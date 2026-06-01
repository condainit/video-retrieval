#!/usr/bin/env python3
"""Rerank top-150 candidates with Qwen3-VL-Reranker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


MODEL = "models/Qwen3-VL-Reranker-8B-pr9"
DEVICE = "cuda"
TOP_K = 150
BATCH_SIZE = 1

VIDEOS_DIR = Path("videos")
LABELS = Path("labels/merged_webvid_ss2.json")
TEST_JSON = Path("data/test-set_no-labels.json")
CANDIDATES_DIR = Path("candidates")
OUTPUT_DIR = Path("reranked")

INSTRUCTION = (
    "Given a reference video, a reference video description, and an edit "
    "instruction, judge whether the candidate video and candidate description "
    "match what the reference video would look like after applying the edit."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def iter_sections(data: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    if isinstance(data, list):
        return [
            (split, rows)
            for section in data
            if isinstance(section, dict)
            for split, rows in section.items()
            if split in {"webvid", "ss2"} and isinstance(rows, list)
        ]

    return [
        (split, rows)
        for split, rows in data.items()
        if split in {"webvid", "ss2"} and isinstance(rows, list)
    ]


def ext_for_split(split: str) -> str:
    if split == "webvid":
        return ".mp4"
    if split == "ss2":
        return ".webm"
    raise ValueError(f"Unknown split: {split}")


def basename_key(value: Any, ext: str) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0]

    text = str(value).strip().replace("\\", "/")
    base = text.split("/")[-1]

    if base.endswith(".mp4") or base.endswith(".webm"):
        return base

    return f"{base}{ext}"


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return value
    return None


def load_label_texts() -> tuple[dict[tuple[str, int], tuple[str, str]], dict[str, str]]:
    query_text: dict[tuple[str, int], tuple[str, str]] = {}
    source_descriptions: dict[str, str] = {}

    for split, rows in iter_sections(read_json(LABELS)):
        ext = ext_for_split(split)

        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                continue

            query_id = int(row["id"])
            modification_text = str(row.get("modification_text", "")).strip()
            description_source = str(row.get("description_source", "")).strip()

            query_text[(split, query_id)] = (modification_text, description_source)

            for field in (
                "video_source",
                "source_video",
                "video_reference",
                "reference_video",
                "source",
                "reference",
                "video",
            ):
                value = row.get(field)
                if value is not None and description_source:
                    source_descriptions.setdefault(
                        basename_key(value, ext),
                        description_source,
                    )

    return query_text, source_descriptions


def load_test_texts() -> tuple[dict[tuple[str, int], tuple[str, str]], dict[tuple[str, int], str]]:
    query_text: dict[tuple[str, int], tuple[str, str]] = {}
    reference_keys: dict[tuple[str, int], str] = {}

    for split, rows in iter_sections(read_json(TEST_JSON)):
        ext = ext_for_split(split)

        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                continue

            query_id = int(row["id"])
            modification_text = str(row.get("modification_text", "")).strip()
            query_text[(split, query_id)] = (modification_text, "")

            source = first_present(
                row,
                (
                    "video_source",
                    "source_video",
                    "video_reference",
                    "reference_video",
                    "reference",
                    "source",
                    "video",
                ),
            )
            reference_keys[(split, query_id)] = basename_key(source, ext)

    return query_text, reference_keys


def load_candidate_keys(npz: Any) -> list[str]:
    if "gallery_keys" in npz:
        return [str(x) for x in npz["gallery_keys"]]
    if "candidate_keys" in npz:
        return [str(x) for x in npz["candidate_keys"]]
    return [
        f"{token}{ext}"
        for token, ext in zip(npz["video_tokens"], npz["gallery_extensions"])
    ]


def load_score_metadata(
    scores_path: Path,
) -> tuple[dict[int, tuple[str, int]], dict[int, str], set[str]]:
    with np.load(scores_path, allow_pickle=True) as z:
        scores = z["scores"]
        gallery_keys = set(load_candidate_keys(z))

        splits = [str(x) for x in z["splits"]]
        query_ids = [int(x) for x in z["query_ids"]]

        if "reference_keys" in z:
            reference_keys = [str(x) for x in z["reference_keys"]]
        else:
            reference_keys = [
                f"{token}{ext}"
                for token, ext in zip(z["reference_tokens"], z["video_extensions"])
            ]

    query_meta = {i: (splits[i], query_ids[i]) for i in range(scores.shape[0])}
    reference_by_query = {i: reference_keys[i] for i in range(scores.shape[0])}

    return query_meta, reference_by_query, gallery_keys


def load_topk(
    topk_path: Path,
    query_meta: dict[int, tuple[str, int]],
) -> dict[int, list[dict[str, Any]]]:
    rows_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with topk_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query_index = int(row["query_index"])
            split, query_id = query_meta[query_index]

            row["query_index"] = query_index
            row["query_id"] = int(row.get("query_id", query_id))
            row["split"] = str(row.get("split", row.get("dataset", split)))
            row["dataset"] = row["split"]
            row["candidate_key"] = str(row["candidate_key"])
            row["baseline_rank"] = int(row.get("baseline_rank", row.get("rank")))

            if "candidate_score" in row and row["candidate_score"] is not None:
                row["candidate_score"] = float(row["candidate_score"])
            elif "score" in row and row["score"] is not None:
                row["candidate_score"] = float(row["score"])

            rows_by_query[query_index].append(row)

    for query_index in rows_by_query:
        rows_by_query[query_index].sort(key=lambda row: row["baseline_rank"])

    return dict(rows_by_query)


def index_videos() -> dict[str, str]:
    return {
        path.name: str(path.resolve())
        for path in VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm"}
    }


def patch_local_model() -> None:
    model_dir = Path(MODEL)
    modules_path = model_dir / "modules.json"

    if not modules_path.exists():
        return

    modules = read_json(modules_path)

    for module in modules:
        if not isinstance(module, dict):
            continue
        module["path"] = str(module.get("path", "")).replace(
            "1_CausalScoreHead",
            "1_LogitScore",
        )
        module["type"] = str(module.get("type", "")).replace(
            "CausalScoreHead",
            "LogitScore",
        )

    write_json(modules_path, modules)

    causal_config = model_dir / "1_CausalScoreHead" / "config.json"
    logit_dir = model_dir / "1_LogitScore"
    logit_config = logit_dir / "config.json"

    if not logit_config.exists() and causal_config.exists():
        logit_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(causal_config, logit_config)

    if logit_config.exists():
        cfg = read_json(logit_config)
        if "true_token_id" in cfg and "false_token_id" in cfg:
            write_json(
                logit_config,
                {
                    "true_token_id": cfg["true_token_id"],
                    "false_token_id": cfg["false_token_id"],
                },
            )


def load_model() -> Any:
    import torch
    from sentence_transformers import CrossEncoder

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    patch_local_model()

    try:
        return CrossEncoder(
            MODEL,
            device=DEVICE,
            model_kwargs={
                "attn_implementation": "sdpa",
                "trust_remote_code": True,
            },
        )
    except TypeError:
        return CrossEncoder(MODEL, device=DEVICE)


def build_query_text(modification_text: str, reference_description: str) -> str:
    return (
        f"{INSTRUCTION}\n\n"
        "Reference video description:\n"
        f"{reference_description if reference_description else '[missing reference description]'}\n\n"
        "Edit instruction:\n"
        f"{modification_text}\n\n"
        "Judge the full video evidence."
    )


def build_document_text(candidate_key: str, candidate_description: str) -> str:
    return (
        "Candidate video key:\n"
        f"{candidate_key}\n\n"
        "Candidate video description:\n"
        f"{candidate_description if candidate_description else '[missing candidate description]'}"
    )


def predict_scores(model: Any, pairs: list[list[dict[str, Any]]]) -> list[float]:
    import torch

    output = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        activation_fct=torch.nn.Sigmoid(),
    )

    arr = np.asarray(output, dtype=np.float32)

    if arr.ndim == 0:
        return [float(arr)]

    if arr.ndim == 1:
        return [float(x) for x in arr.tolist()]

    if arr.shape[1] == 1:
        return [float(x) for x in arr[:, 0].tolist()]

    return [float(x) for x in arr[:, -1].tolist()]


def rerank_query(
    model: Any,
    entries: list[dict[str, Any]],
    query_text_map: dict[tuple[str, int], tuple[str, str]],
    source_descriptions: dict[str, str],
    reference_by_query: dict[int, str],
    video_index: dict[str, str],
) -> list[dict[str, Any]]:
    first = entries[0]
    query_index = int(first["query_index"])
    split = first["split"]
    query_id = int(first["query_id"])

    modification_text, reference_description = query_text_map[(split, query_id)]
    reference_key = reference_by_query[query_index]
    reference_video = video_index[reference_key]

    query_text = build_query_text(modification_text, reference_description)

    rows: list[dict[str, Any]] = []
    pairs: list[list[dict[str, Any]]] = []

    for row in entries[:TOP_K]:
        candidate_key = row["candidate_key"]

        if not candidate_key.endswith(ext_for_split(split)):
            continue

        candidate_video = video_index[candidate_key]
        candidate_description = source_descriptions.get(candidate_key, "")
        document_text = build_document_text(candidate_key, candidate_description)

        pairs.append(
            [
                {
                    "text": query_text,
                    "video": os.path.abspath(reference_video),
                },
                {
                    "text": document_text,
                    "video": os.path.abspath(candidate_video),
                },
            ]
        )
        rows.append(row)

    scores = predict_scores(model, pairs)

    for row, score in zip(rows, scores):
        row["vl_video_reranker_score"] = float(score)
        row["vl_video_reranker_model"] = MODEL
        row["vl_video_reranker_backend"] = "sentence_transformers_cross_encoder_fullvideo"
        row["reference_key"] = reference_key
        row["reference_video_path"] = reference_video
        row["candidate_video_path"] = video_index[row["candidate_key"]]

    return rows


def main() -> int:
    args = parse_args()

    scores_path = CANDIDATES_DIR / args.split / "scores.npz"
    topk_path = CANDIDATES_DIR / args.split / f"topk{TOP_K}_candidates.jsonl"
    output_path = OUTPUT_DIR / args.split / "scores.jsonl"

    query_text_map, source_descriptions = load_label_texts()
    query_meta, reference_by_query, gallery_keys = load_score_metadata(scores_path)

    if args.split == "test":
        test_query_text, test_reference_keys = load_test_texts()
        query_text_map.update(test_query_text)

        for query_index, meta in query_meta.items():
            reference_by_query[query_index] = test_reference_keys.get(
                meta,
                reference_by_query[query_index],
            )

    topk = load_topk(topk_path, query_meta)
    video_index = index_videos()
    model = load_model()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_indices = sorted(topk)

    with output_path.open("w", encoding="utf-8") as f:
        for query_index in tqdm(
            query_indices,
            desc=f"Reranking {args.split}",
            unit="query",
        ):
            entries = [
                row
                for row in topk[query_index][:TOP_K]
                if row["candidate_key"] in gallery_keys
            ]

            if not entries:
                continue

            for row in rerank_query(
                model=model,
                entries=entries,
                query_text_map=query_text_map,
                source_descriptions=source_descriptions,
                reference_by_query=reference_by_query,
                video_index=video_index,
            ):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
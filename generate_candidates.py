#!/usr/bin/env python3
"""Generate top-150 embedding candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEVICE = "cuda"
TOP_K = 150

VIDEOS_DIR = Path("videos")
GALLERY_EMBEDDINGS = Path("embeddings/gallery_embeddings.npz")
TEST_JSON = Path("data/test-set_no-labels.json")
VALIDATION_JSON = Path("labels/merged_webvid_ss2.json")
OUTPUT_ROOT = Path("candidates")

QUERY_PROMPT = (
    "Given the reference video and edit instruction, represent the target video "
    "that would result after applying the edit.\n\n"
    "Edit instruction: {modification_text}"
)
ENCODE_PROMPT = "Represent the user's input."


@dataclass
class Query:
    query_index: int
    split: str
    query_id: int
    modification_text: str
    reference_key: str
    reference_path: str
    target_key: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return value
    return None


def basename_key(value: Any, ext: str) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0]

    text = str(value).strip().replace("\\", "/")
    base = text.split("/")[-1]

    if base.endswith(".mp4") or base.endswith(".webm"):
        return base

    return f"{base}{ext}"


def infer_reference_key(row: dict[str, Any], split: str) -> str:
    value = first_present(
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
    if value is None:
        raise RuntimeError(f"Missing reference video for split={split} id={row.get('id')}")
    return basename_key(value, ext_for_split(split))


def infer_target_key(row: dict[str, Any], split: str) -> str:
    value = first_present(
        row,
        (
            "video_target",
            "target_video",
            "target",
            "video_target_id",
            "target_video_id",
            "video_destination",
        ),
    )
    return "" if value is None else basename_key(value, ext_for_split(split))


def load_queries(path: Path, include_targets: bool) -> list[Query]:
    queries: list[Query] = []

    for split, rows in iter_sections(read_json(path)):
        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                continue

            modification_text = str(row.get("modification_text", "")).strip()
            if not modification_text:
                raise RuntimeError(f"Missing modification_text for split={split} id={row.get('id')}")

            reference_key = infer_reference_key(row, split)
            reference_path = VIDEOS_DIR / reference_key
            if not reference_path.exists():
                raise RuntimeError(f"Missing reference video: {reference_path}")

            queries.append(
                Query(
                    query_index=len(queries),
                    split=split,
                    query_id=int(row["id"]),
                    modification_text=modification_text,
                    reference_key=reference_key,
                    reference_path=str(reference_path.resolve()),
                    target_key=infer_target_key(row, split) if include_targets else "",
                )
            )

    if not queries:
        raise RuntimeError(f"No queries loaded from {path}")

    return queries


def load_model() -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    return SentenceTransformer(MODEL, device=DEVICE)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def load_gallery_embeddings() -> tuple[np.ndarray, list[str]]:
    with np.load(GALLERY_EMBEDDINGS, allow_pickle=True) as z:
        embeddings = z["embeddings"].astype(np.float32)
        gallery_keys = [str(x) for x in z["gallery_keys"]]

    return l2_normalize(embeddings), gallery_keys


def encode_queries(model: Any, queries: list[Query], split_name: str) -> np.ndarray:
    outputs: list[np.ndarray] = []

    for query in tqdm(queries, desc=f"Encoding {split_name}", unit="query"):
        emb = model.encode(
            [
                {
                    "text": QUERY_PROMPT.format(
                        modification_text=query.modification_text,
                    ),
                    "video": query.reference_path,
                }
            ],
            prompt=ENCODE_PROMPT,
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        arr = emb.detach().float().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[None, :]

        outputs.append(arr.astype(np.float32, copy=False))

    return l2_normalize(np.vstack(outputs).astype(np.float32, copy=False))


def same_split(key: str, split: str) -> bool:
    return key.endswith(ext_for_split(split))


def topk_indices(scores: np.ndarray, query: Query, gallery_keys: list[str]) -> list[int]:
    selected: list[int] = []

    for gi in np.argsort(-scores[query.query_index]):
        candidate_key = gallery_keys[int(gi)]

        if candidate_key == query.reference_key:
            continue
        if not same_split(candidate_key, query.split):
            continue

        selected.append(int(gi))

        if len(selected) == TOP_K:
            break

    return selected


def write_scores_npz(
    path: Path,
    scores: np.ndarray,
    queries: list[Query],
    gallery_keys: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        query_indices=np.asarray([q.query_index for q in queries], dtype=np.int32),
        query_ids=np.asarray([q.query_id for q in queries], dtype=np.int32),
        splits=np.asarray([q.split for q in queries], dtype=object),
        reference_tokens=np.asarray([Path(q.reference_key).stem for q in queries], dtype=object),
        video_extensions=np.asarray([Path(q.reference_key).suffix for q in queries], dtype=object),
        reference_keys=np.asarray([q.reference_key for q in queries], dtype=object),
        target_keys=np.asarray([q.target_key for q in queries], dtype=object),
        gallery_keys=np.asarray(gallery_keys, dtype=object),
    )


def write_topk_jsonl(
    path: Path,
    scores: np.ndarray,
    queries: list[Query],
    gallery_keys: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with path.open("w", encoding="utf-8") as f:
        for query in queries:
            for rank, gi in enumerate(topk_indices(scores, query, gallery_keys), start=1):
                row = {
                    "query_index": query.query_index,
                    "query_id": query.query_id,
                    "split": query.split,
                    "rank": rank,
                    "candidate_index": gi,
                    "candidate_key": gallery_keys[gi],
                    "score": float(scores[query.query_index, gi]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1

    print(f"Wrote {path}")
    print(f"Rows: {rows}")


def main() -> int:
    args = parse_args()

    input_json = TEST_JSON if args.split == "test" else VALIDATION_JSON
    output_dir = OUTPUT_ROOT / args.split

    gallery_embeddings, gallery_keys = load_gallery_embeddings()
    queries = load_queries(input_json, include_targets=args.split == "validation")

    query_embeddings = encode_queries(load_model(), queries, args.split)
    scores = (query_embeddings @ gallery_embeddings.T).astype(np.float32)

    write_scores_npz(output_dir / "scores.npz", scores, queries, gallery_keys)
    write_topk_jsonl(output_dir / f"topk{TOP_K}_candidates.jsonl", scores, queries, gallery_keys)

    print(f"Wrote {output_dir / 'scores.npz'}")
    print(f"Queries: {len(queries)}")
    print(f"Gallery videos: {len(gallery_keys)}")
    print(f"Score shape: {scores.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
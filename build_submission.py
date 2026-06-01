#!/usr/bin/env python3
"""Build a submission."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


LABELS = Path("labels/merged_webvid_ss2.json")
TEST_JSON = Path("data/test-set_no-labels.json")
VIDEOS_DIR = Path("videos")
CANDIDATES_DIR = Path("candidates")
FUSION_DIR = Path("fusion")
SUBMISSIONS_DIR = Path("submissions")

TOP_K = 50

TRACE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TRACE_FPS = 1.0
TRACE_MAX_NEW_TOKENS = 160


@dataclass
class QueryMeta:
    query_index: int
    query_id: int
    split: str
    reference_token: str


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


def maybe_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def build_webvid_map() -> dict[str, str]:
    mapping: dict[str, str] = {}

    for split, rows in iter_sections(read_json(LABELS)):
        if split != "webvid":
            continue

        for row in rows:
            for field in ("video_source", "video_target"):
                value = row.get(field)
                if value is None:
                    continue

                full_id = str(value).strip().replace("\\", "/")
                bare_id = Path(full_id).name

                if bare_id.endswith(".mp4"):
                    bare_id = bare_id[:-4]
                if full_id.endswith(".mp4"):
                    full_id = full_id[:-4]

                mapping[bare_id] = full_id

    return mapping


def load_query_meta(scores_path: Path) -> list[QueryMeta]:
    with np.load(scores_path, allow_pickle=True) as z:
        splits = [str(x) for x in z["splits"]]
        query_ids = [int(x) for x in z["query_ids"]]

        if "reference_keys" in z:
            reference_tokens = [Path(str(x)).stem for x in z["reference_keys"]]
        else:
            reference_tokens = [str(x) for x in z["reference_tokens"]]

    return [
        QueryMeta(
            query_index=i,
            query_id=query_ids[i],
            split=splits[i],
            reference_token=reference_tokens[i],
        )
        for i in range(len(query_ids))
    ]


def load_fused_rows(path: Path) -> dict[int, list[str]]:
    rows: dict[int, list[tuple[int, str]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.setdefault(int(row["query_index"]), []).append(
                (int(row["rank"]), str(row["candidate_key"]))
            )

    return {
        query_index: [candidate_key for _rank, candidate_key in sorted(values)]
        for query_index, values in rows.items()
    }


def map_webvid_id(video_key: str, webvid_map: dict[str, str]) -> str:
    bare_id = Path(video_key).stem
    return webvid_map.get(bare_id, bare_id)


def map_video_id(video_key: str, split: str, webvid_map: dict[str, str]) -> int | str:
    bare_id = Path(video_key).stem

    if split == "webvid":
        return map_webvid_id(video_key, webvid_map)

    return maybe_int(bare_id)


def build_submission(
    query_meta: list[QueryMeta],
    fused_rows: dict[int, list[str]],
    webvid_map: dict[str, str],
) -> list[dict[str, list[dict[str, Any]]]]:
    output: dict[str, list[dict[str, Any]]] = {"webvid": [], "ss2": []}

    for meta in query_meta:
        candidate_keys = fused_rows[meta.query_index][:TOP_K]

        if meta.split == "webvid":
            video_source: int | str = webvid_map.get(
                meta.reference_token,
                meta.reference_token,
            )
        else:
            video_source = maybe_int(meta.reference_token)

        entry = {
            "id": meta.query_id,
            "video_source": video_source,
            "video_target": [
                map_video_id(candidate_key, meta.split, webvid_map)
                for candidate_key in candidate_keys
            ],
        }

        output[meta.split].append(entry)

    return [{"webvid": output["webvid"]}, {"ss2": output["ss2"]}]


def load_test_modifications() -> dict[tuple[str, int], str]:
    modifications: dict[tuple[str, int], str] = {}

    for split, rows in iter_sections(read_json(TEST_JSON)):
        for row in rows:
            modifications[(split, int(row["id"]))] = str(row["modification_text"]).strip()

    return modifications


def index_videos() -> dict[str, Path]:
    lookup: dict[str, Path] = {}

    for path in VIDEOS_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".webm"}:
            continue

        resolved = path.resolve()
        rel = str(path.relative_to(VIDEOS_DIR)).replace("\\", "/")
        rel_no_ext = str(Path(rel).with_suffix("")).replace("\\", "/")

        lookup[path.name] = resolved
        lookup[path.stem] = resolved
        lookup[rel] = resolved
        lookup[rel_no_ext] = resolved

    return lookup


def normalize_video_id(value: Any) -> str:
    text = str(value).strip().replace("\\", "/").strip("/")
    return re.sub(r"\.(mp4|webm)$", "", text, flags=re.IGNORECASE)


def resolve_video(value: Any, split: str, video_lookup: dict[str, Path]) -> Path:
    raw = normalize_video_id(value)
    name = Path(raw).name
    ext = ".mp4" if split == "webvid" else ".webm"

    for key in (raw, name, f"{raw}{ext}", f"{name}{ext}"):
        if key in video_lookup:
            return video_lookup[key]

    return video_lookup[name]


def build_trace_prompt(modification_text: str) -> str:
    return (
        "You are generating a reasoning trace for a composed video retrieval prediction.\n\n"
        "You are given two videos:\n"
        "1. The reference video before the edit.\n"
        "2. The predicted target video after the edit.\n\n"
        "Write a concise reasoning trace explaining why the predicted target video matches "
        "the result of applying the edit to the reference video.\n\n"
        "Focus on observable visual changes:\n"
        "- subject or object state changes\n"
        "- actions or temporal phases\n"
        "- scene or background changes\n"
        "- camera/framing changes\n"
        "- motion, pacing, or atmosphere\n\n"
        "Do not mention uncertainty.\n"
        "Do not mention ranking, retrieval, score, dataset, or prediction.\n"
        'Do not say "I cannot see" or "the model predicts."\n'
        "Do not invent details that are not visible.\n"
        "Write one compact paragraph, 3-5 sentences.\n\n"
        f"Edit instruction:\n{modification_text}"
    )


def clean_trace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^reasoning trace\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^trace\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) > 180:
        text = " ".join(words[:180]).rstrip(" ,;:") + "."

    return text


def load_trace_model() -> tuple[Any, Any, Any]:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(TRACE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TRACE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, processor, process_vision_info


def generate_trace(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    source_video: Path,
    target_video: Path,
    modification_text: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reference video before the edit:"},
                {
                    "type": "video",
                    "video": source_video.resolve().as_uri(),
                    "fps": TRACE_FPS,
                },
                {"type": "text", "text": "Predicted target video after the edit:"},
                {
                    "type": "video",
                    "video": target_video.resolve().as_uri(),
                    "fps": TRACE_FPS,
                },
                {"type": "text", "text": build_trace_prompt(modification_text)},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=TRACE_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return clean_trace(decoded)


def add_reasoning_traces(
    submission: list[dict[str, list[dict[str, Any]]]],
) -> list[dict[str, list[dict[str, Any]]]]:
    modifications = load_test_modifications()
    video_lookup = index_videos()
    model, processor, process_vision_info = load_trace_model()

    for split, rows in iter_sections(submission):
        for row in tqdm(rows, desc=f"Reasoning traces {split}", unit="query"):
            source_video = resolve_video(row["video_source"], split, video_lookup)
            target_video = resolve_video(row["video_target"][0], split, video_lookup)
            modification_text = modifications[(split, int(row["id"]))]

            row["reasoning_trace"] = [
                generate_trace(
                    model=model,
                    processor=processor,
                    process_vision_info=process_vision_info,
                    source_video=source_video,
                    target_video=target_video,
                    modification_text=modification_text,
                )
            ]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return submission


def main() -> int:
    args = parse_args()

    scores_path = CANDIDATES_DIR / args.split / "scores.npz"
    fused_path = FUSION_DIR / args.split / f"fused_top{TOP_K}.jsonl"
    submission_path = SUBMISSIONS_DIR / args.split / "submission.json"

    submission = build_submission(
        query_meta=load_query_meta(scores_path),
        fused_rows=load_fused_rows(fused_path),
        webvid_map=build_webvid_map(),
    )

    if args.split == "test":
        submission = add_reasoning_traces(submission)

    write_json(submission_path, submission)
    print(f"Wrote {submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
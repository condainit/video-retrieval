#!/usr/bin/env python3
"""Fuse embedding and reranker scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATES_DIR = Path("candidates")
RERANKED_DIR = Path("reranked")
FUSION_DIR = Path("fusion")

TOP_K = 150
OUTPUT_TOP_K = 50
RERANKER_WEIGHT = 0.65
RERANKER_SCORE_FIELD = "vl_video_reranker_score"


@dataclass
class QueryMeta:
    query_index: int
    query_id: int
    split: str
    reference_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    return parser.parse_args()


def suffix_for_split(split: str) -> str:
    if split == "webvid":
        return ".mp4"
    if split == "ss2":
        return ".webm"
    raise ValueError(f"Unknown split: {split}")


def load_candidate_keys(npz: Any) -> list[str]:
    if "gallery_keys" in npz:
        return [str(x) for x in npz["gallery_keys"]]
    if "candidate_keys" in npz:
        return [str(x) for x in npz["candidate_keys"]]
    return [
        f"{token}{ext}"
        for token, ext in zip(npz["video_tokens"], npz["gallery_extensions"])
    ]


def load_scores(scores_path: Path) -> tuple[np.ndarray, list[str], list[QueryMeta]]:
    with np.load(scores_path, allow_pickle=True) as z:
        scores = z["scores"].astype(np.float32)
        candidate_keys = load_candidate_keys(z)

        splits = [str(x) for x in z["splits"]]
        query_ids = [int(x) for x in z["query_ids"]]

        if "reference_keys" in z:
            reference_keys = [str(x) for x in z["reference_keys"]]
        else:
            reference_keys = [
                f"{token}{ext}"
                for token, ext in zip(z["reference_tokens"], z["video_extensions"])
            ]

    query_meta = [
        QueryMeta(
            query_index=i,
            query_id=query_ids[i],
            split=splits[i],
            reference_key=reference_keys[i],
        )
        for i in range(len(query_ids))
    ]

    return scores, candidate_keys, query_meta


def load_reranker_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            query_index = int(row["query_index"])
            row["query_index"] = query_index
            row["candidate_key"] = str(row["candidate_key"])
            row["baseline_rank"] = int(row.get("baseline_rank", row.get("rank")))
            row[RERANKER_SCORE_FIELD] = float(row[RERANKER_SCORE_FIELD])

            if "candidate_score" in row and row["candidate_score"] is not None:
                row["candidate_score"] = float(row["candidate_score"])
            elif "score" in row and row["score"] is not None:
                row["candidate_score"] = float(row["score"])

            rows.setdefault(query_index, []).append(row)

    for query_index in rows:
        rows[query_index].sort(key=lambda row: row["baseline_rank"])

    return rows


def base_score(
    row: dict[str, Any],
    scores: np.ndarray,
    key_to_index: dict[str, int],
) -> float:
    if "candidate_score" in row and row["candidate_score"] is not None:
        return float(row["candidate_score"])

    return float(scores[int(row["query_index"]), key_to_index[row["candidate_key"]]])


def fuse(base: float, reranker: float) -> float:
    return float((1.0 - RERANKER_WEIGHT) * base + RERANKER_WEIGHT * reranker)


def fuse_query(
    meta: QueryMeta,
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    candidate_keys: list[str],
    key_to_index: dict[str, int],
) -> list[dict[str, Any]]:
    fused_rows: list[dict[str, Any]] = []
    expected_suffix = suffix_for_split(meta.split)

    for row in rows[:TOP_K]:
        candidate_key = row["candidate_key"]

        if candidate_key == meta.reference_key:
            continue
        if not candidate_key.endswith(expected_suffix):
            continue
        if candidate_key not in key_to_index:
            continue

        b = base_score(row, scores, key_to_index)
        r = float(row[RERANKER_SCORE_FIELD])

        fused_rows.append(
            {
                "query_index": meta.query_index,
                "query_id": meta.query_id,
                "split": meta.split,
                "candidate_key": candidate_key,
                "fused_score": fuse(b, r),
                "source": "reranked",
            }
        )

    fused_rows.sort(key=lambda row: row["fused_score"], reverse=True)

    used = {row["candidate_key"] for row in fused_rows}
    ranked = np.argsort(-scores[meta.query_index])

    for gallery_index in ranked:
        if len(fused_rows) >= OUTPUT_TOP_K:
            break

        candidate_key = candidate_keys[int(gallery_index)]

        if candidate_key == meta.reference_key:
            continue
        if candidate_key in used:
            continue
        if not candidate_key.endswith(expected_suffix):
            continue

        fused_rows.append(
            {
                "query_index": meta.query_index,
                "query_id": meta.query_id,
                "split": meta.split,
                "candidate_key": candidate_key,
                "fused_score": float(scores[meta.query_index, int(gallery_index)]),
                "source": "base_fill",
            }
        )
        used.add(candidate_key)

    for rank, row in enumerate(fused_rows[:OUTPUT_TOP_K], start=1):
        row["rank"] = rank

    return fused_rows[:OUTPUT_TOP_K]


def main() -> int:
    args = parse_args()

    scores_path = CANDIDATES_DIR / args.split / "scores.npz"
    reranked_path = RERANKED_DIR / args.split / "scores.jsonl"
    output_path = FUSION_DIR / args.split / f"fused_top{OUTPUT_TOP_K}.jsonl"

    scores, candidate_keys, query_meta = load_scores(scores_path)
    reranker_rows = load_reranker_rows(reranked_path)
    key_to_index = {key: i for i, key in enumerate(candidate_keys)}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for meta in query_meta:
            rows = fuse_query(
                meta=meta,
                rows=reranker_rows.get(meta.query_index, []),
                scores=scores,
                candidate_keys=candidate_keys,
                key_to_index=key_to_index,
            )

            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build Qwen3-VL gallery embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


MODEL = "Qwen/Qwen3-VL-Embedding-8B"
VIDEOS_DIR = Path("videos")
OUTPUT = Path("embeddings/gallery_embeddings.npz")
DEVICE = "cuda"

GALLERY_TEXT = "Represent this video for retrieval."
ENCODE_PROMPT = "Represent the user's input."
VIDEO_EXTENSIONS = {".mp4", ".webm"}


def split_from_key(key: str) -> str:
    if key.endswith(".mp4"):
        return "webvid"
    if key.endswith(".webm"):
        return "ss2"
    raise ValueError(f"Unsupported video key: {key}")


def list_gallery_videos() -> list[Path]:
    video_paths = sorted(
        path.resolve()
        for path in VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise RuntimeError(f"No gallery videos found in {VIDEOS_DIR}")
    return video_paths


def load_model() -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    return SentenceTransformer(MODEL, device=DEVICE)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def encode_gallery(model: Any, video_paths: list[Path]) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for path in tqdm(video_paths, desc="Encoding gallery", unit="video"):
            emb = model.encode(
                [{"text": GALLERY_TEXT, "video": str(path)}],
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


def save_embeddings(embeddings: np.ndarray, video_paths: list[Path]) -> None:
    gallery_keys = [path.name for path in video_paths]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        embeddings=embeddings.astype(np.float32),
        gallery_keys=np.asarray(gallery_keys, dtype=object),
        gallery_paths=np.asarray([str(path) for path in video_paths], dtype=object),
        gallery_splits=np.asarray(
            [split_from_key(key) for key in gallery_keys],
            dtype=object,
        ),
    )


def main() -> int:
    video_paths = list_gallery_videos()
    embeddings = encode_gallery(load_model(), video_paths)
    save_embeddings(embeddings, video_paths)

    print(f"Wrote {OUTPUT}")
    print(f"Gallery videos: {len(video_paths)}")
    print(f"Embedding shape: {embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
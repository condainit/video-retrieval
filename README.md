# Embedding Retrieval and Video Reranking for Reason-Aware Composed Video Retrieval

## Multimodal Video Intelligence Team

## Quick Start
Python 3.11 is recommended.

The commands below default to the test split. To run the same pipeline on the validation split, add `--split validation` to Steps 3–6.

### 1. Setup

a. Install PyTorch for your CUDA version: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

b. Install the remaining requirements: 

```bash 
pip install -r requirements.txt
```
c. Download the dataset and prepare the flat video directory:

```bash
hf download orange-fox/CoVR-R \
  --repo-type dataset \
  --local-dir data \
  --force-download

mkdir -p videos && find data -type f \( -name "*.mp4" -o -name "*.webm" \) -exec cp {} videos/ \;
```

d. Download `Qwen3-VL-Reranker-8B`:

```bash
mkdir -p models

hf download Qwen/Qwen3-VL-Reranker-8B \
  --local-dir models/Qwen3-VL-Reranker-8B-pr9 \
  --revision refs/pr/9
```

e. Place the official `merged_webvid_ss2.json` labels file from [mbzuai-oryx/CoVR-R](https://github.com/mbzuai-oryx/CoVR-R) at `labels/merged_webvid_ss2.json`.

### 2. Build Gallery Embeddings

Encode all gallery videos once with `Qwen/Qwen3-VL-Embedding-8B`.

```bash
python build_gallery_embeddings.py
```

### 3. Generate Retrieval Candidates

Generate top-150 same-split retrieval candidates with `Qwen/Qwen3-VL-Embedding-8B`.

```bash
python generate_candidates.py
```

Note: This step regenerates query embeddings and top-150 nearest-neighbor candidates. Small numerical differences across environments can change candidates near the top-150 boundary.

### 4. Rerank Candidates

Rerank the top-150 candidates with `Qwen3-VL-Reranker-8B` using full reference and candidate videos.

```bash
python rerank_candidates.py
```

### 5. Fuse Scores

Fuse retrieval and reranker scores with weight `0.65`.

```bash
python fuse_scores.py
```

### 6. Build Submission

Build the final challenge-format submission.

```bash
python build_submission.py
```
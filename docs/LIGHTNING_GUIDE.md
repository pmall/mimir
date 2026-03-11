# ⚡ Lightning AI - Terminal Training Guide

Follow these steps to run your optimized ESM-3 training on a **Lightning Studio** (H100 or A100).

## 1. Environment Setup

Studios come with `uv` and `git` pre-installed. Run these in the Studio terminal:

```bash
# Clone and enter repo
git clone https://github.com/pmall/mimir.git
cd mimir

# Sync dependencies (Strict uv environment)
uv sync
```

## 2. Prepare Dataset

Mimir v2 requires a unified structured dataset with pre-processed LMDBs and a centralized `config.json`.

1. Upload your dataloader directory (e.g., `data/run78-v2/`) into the `data/` folder in the file explorer sidebar.
2. This directory must contain your `config.json`, the fingerprints LMDB, and binders LMDB.

## 3. Model & Auth

You must authorize with Hugging Face to download the ESM-3 weights.

```bash
hf auth login
uv run scripts/download_weights.py
```

## 4. VRAM Crash Testing

Before kicking off a long training run, test your hardware limits to find the maximum VRAM-safe batch size:

```bash
uv run python -m scripts.train_crash_test --batch-size 128 --accum 1
```

_Lower `--batch-size` until it completes without OOM. Use the maximum successful value in the next step._

## 5. Training

```bash
uv run python -m scripts.train \
  --config data/run78-v2/config.json \
  --checkpoint-dir runs/run2 \
  --epochs 500 \
  --batch-size <YOUR_MAX_BATCH> \
  --gradient-accumulation-steps <CALCULATED_ACCUM> \
  --lam 0.25 \
  --use-8bit-adam \
  --peak-lr 1e-4
```

> **Note:** The first epoch takes longer due to `torch.compile` graph tracing. Subsequent epochs are faster.

**Estimated time** (500 epochs):

- H100 80GB: ~30-40 hours
- A100 80GB: ~65-80 hours

## 6. Download Results

```bash
# Zip the entire run folder
zip -r mimir_run2.zip runs/run2/
```

_Right-click `mimir_run2.zip` in the VS Code sidebar and select **Download**._

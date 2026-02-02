# ⚡ Lightning AI - Terminal Training Guide

Follow these steps to run your optimized ESM-3 training on a **Lightning Studio** (H100 or A100).

## 1. Environment Setup

Studios come with `uv` and `git` pre-installed. Run these in the Studio terminal:

```bash
# Clone and enter repo
git clone https://github.com/pmall/mimir.git
cd mimir

# Install dependencies (Editable mode)
pip install -e .
```

## 2. Upload Dataset

You need to provide a dataset of peptide sequences and their target proteins.

1.  **Format**: A CSV file with two columns: `sequence` and `target`.
    ```csv
    sequence,target
    MKTIIALSYIFCLVF,ProteinA
    ACDEFGHIKLMNPQR,ProteinB
    ...
    ```
2.  **Upload**: Drag and drop your `.csv` file into the `data/` folder in the file explorer sidebar.

    > [!NOTE]
    > If you don't have a dataset, you can use the default `data/mapping_dataset.csv`.

## 3. Model & Auth

You must authorize with Hugging Face to download the ESM-3 weights.

```bash
huggingface-cli login
python scripts/download_weights.py
```

## 4. High-Performance Training

Use the following command for an **H100 (80GB)** or **A100 (80GB)**. This uses a large batch size and BF16 for maximum speed.

```bash
# Fix segment fragmentation (set both to ensure compatibility)
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run Training
uv run scripts/train.py \
  --dataset data/mapping_dataset.csv \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --use_8bit_adam \
  --epochs 100 \
  --lr 1e-4 2>&1 | tee >(grep --line-buffered -v "it/s" > training_log.txt)
```

**Total Time Estimation**:

- H100: ~6-8 hours for 100 epochs.
- A100 (80GB): ~13-16 hours for 100 epochs.

## 5. Manual Snapshots (Optional)

Since the script only saves the `best_model` and `last_model`, you might want to manually save a specific epoch (e.g., Epoch 10) while training continues.

1.  Open a **new terminal** in the Studio.
2.  Copy the current state to a new folder and zip it:

    ```bash
    # Example: Save current state as epoch_10
    cp -r checkpoints/last_model checkpoints/epoch_10
    zip -r mimir_epoch_10.zip checkpoints/epoch_10
    ```

## 6. Download Results

Once training finishes, the best model is saved in `checkpoints/best_model`.

```bash
# Zip for downloading
zip -r mimir_best_model.zip checkpoints/best_model
```

_Right-click `mimir_best_model.zip` in the VS Code sidebar and select **Download**._

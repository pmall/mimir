# MÍMIR

**MÍMIR** is a generative AI project designed to generate peptides that bind to specific human protein targets.

## The Concept

Designing proteins that bind to specific targets is a complex biological puzzle. In the past, this required checking billions of random combinations or training models from scratch on limited data. **MÍMIR** takes a different approach: **"Steering a Giant."**

### 1. The Engine: ESM-3

We are not building a brain from scratch. We are using **ESM-3**, a state-of-the-art "Large Protein Model" trained on billions of evolutionary sequences. ESM-3 already understands the fundamental laws of biology—how proteins fold, interact, and function—much like how GPT-4 understands grammar and logic.

### 2. The Conditioning: Dealing with "Who?"

ESM-3 is a generalist; it knows how to make proteins, but not _which_ protein you need right now. To solve this, we introduce **Target Conditioning**.

- We assign a unique digital "ID card" (a special token) to every human target protein (e.g., `<TARGET_P53>`).
- We prompt the model with this token: _"Given `<TARGET_P53>`, complete the rest of this sequence."_

### 3. The Fine-Tuning: LoRA

We use **LoRA (Low-Rank Adaptation)** to fine-tune the model. Instead of rewriting the entire model (which is prohibitively expensive), we insert small, trainable "adapters" into its attention layers.

- These adapters learn to associate the "ID card" with specific chemical properties needed for binding.
- This allows us to leverage the massive general knowledge of ESM-3 while steering it to focus specifically on target binding affinity.

In essence, we are taking a "Universe Simulator" of proteins and tweaking its settings so that its "Create Protein" button defaults to "Create Peptide Binder" for your specific target.

## Directory Structure

- **`mimir/`**: Core package containing dataset and tokenizer logic.
  - `dataset.py`: PyTorch Dataset implementation with dynamic padding.
  - `tokenizer.py`: Wrapper around ESM-3 tokenizer.
- **`scripts/`**: Executable scripts for data generation and training.
  - `generate_dataset.py`: Extracts interacting peptide pairs from the PostgreSQL database.
  - `train.py`: Fine-tunes ESM-3 using LoRA.
- **`data/`**: Ignored directory where generated datasets are stored.

## Setup

This project uses `uv` for dependency management.

1.  **Initialize environment**:

    ```bash
    uv sync
    ```

2.  **Configure Environment**:
    Create a `.env` file with optimization for your database:
    ```
    POSTGRES_HOST=...
    POSTGRES_PORT=...
    POSTGRES_DB=...
    POSTGRES_USER=...
    POSTGRES_PASSWORD=...
    ```

## Usage

### 1. Generate Dataset

Extract the interacting peptide sequences from the database. This script generates `data/dataset.csv`.

```bash
uv run python scripts/generate_dataset.py --verbose
```

### 2. Fine-tune ESM-3

Train the model using LoRA.

```bash
uv run python scripts/train.py --epochs 10 --batch_size 4
```

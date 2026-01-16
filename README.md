# MÍMIR v2

**MÍMIR v2** is a project for generating viral peptides using the ESM-3 protein language model. It leverages LoRA (Low-Rank Adaptation) fine-tuning to steer the model towards generating specific viral sequences that interact with human targets.

## Project Goal

The primary objective is to transition from training models from scratch to "steering a giant". By fine-tuning the state-of-the-art **ESM-3** model, we aim to generate high-quality viral binders for specific human protein targets.

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

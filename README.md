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

### 4. Implementation Details

We implement this strategy using a **Masked Language Modeling (MLM)** objective, similar to how BERT is trained, but adapted for generative purpose:

1.  **Target Conditioning**:

    - Every training sample is prepended with its target ID: `[TARGET_ID] [BOS] [SEQUENCE...] [EOS]`
    - This creates a strong association: The model learns that _this_ specific ID implies _this_ style of sequence.

2.  **Masking Strategy**:

    - We apply **variable random masking (15% - 50%)** to sequence tokens.
    - **Why?** Uniform 15% masking provides insufficient training signal across all sequence lengths for this generative task. Variable higher masking forces the model to reconstruct larger portions of the structure and learn more robust internal representations.
    - **Smart Masking**: We ensure at least one mask is always applied.
    - **Crucial**: The `TARGET_ID` is **never masked**. It is always visible to condition the repair process.

3.  **Training Objective**:

    - The model predicts the missing amino acids based on the visible ones AND the target ID.
    - Loss is calculated _only_ on the masked positions, normalized by the number of masks.

4.  **Technical Nuances**:
    - **Embedding Resizing**: Since ESM-3 uses a fixed vocabulary, we manually resize the embedding layers to accommodate our new target tokens.
    - **LoRA Targets**: We fine-tune specific attention modules (`layernorm_qkv` and `out_proj`) to efficiently adapt the model's "grammar" to our specific "dialect" of binders.

## Directory Structure

- **`data/`**: Ignored directory where generated datasets are stored.
- **`mimir/`**: Core package containing dataset and tokenizer logic.
  - `dataset.py`: PyTorch Dataset implementation with dynamic padding.
  - `model_utils.py`: Utilities for resizing ESM-3 embeddings.
  - `tokenizer.py`: Wrapper around ESM-3 tokenizer.
- **`scripts/`**: Executable scripts for data generation and training.
  - `download_weights.py`: Triggers model weight download via `esm` library.
  - `estimate_training.py`: Calculates sample complexity and estimates training time.
  - `generate_dataset.py`: Extracts interacting peptide pairs from the PostgreSQL database.
  - `test_esm3.py`: Validates installation by running a simple generation task.
  - `train.py`: Fine-tunes ESM-3 using LoRA.
- **`setup_esm3.sh`**: Environment setup, authentication, and weight download.

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
uv run scripts/train.py --epochs 500 --batch_size 4
```

### 3. Estimate Training Resources

We provide a script to estimate the combinatorial complexity and training time based on your dataset statistics and masking strategy.

```bash
uv run scripts/estimate_training.py
```

**Estimated Training Times (500 Epochs):**

- **Google Colab (T4 GPU)**: ~3.5 hours
- **Google Colab (H100 GPU)**: ~25 minutes

_Note: 500 epochs provide robust coverage for short peptides while randomly sampling the massive combinatorial space of longer sequences._

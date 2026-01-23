# MÍMIR

**MÍMIR** is a specialized framework for **de novo peptide design**.

By leveraging the generative capabilities of **ESM-3**, MÍMIR outputs novel peptide sequences (< 20 amino acids) conditioned to bind specific human protein targets. It transforms the problem of finding a binder from a random search into a targeted generation task.

## The Concept

Finding effective peptide binders for specific protein targets is a challenge of **combinatorial biology**. The number of possible peptide sequences is astronomical (20^sequence_length), making random screening inefficient.

**MÍMIR** solves this by treating biology as a language.

### 1. The Foundation: ESM-3

We leverage **ESM-3**, a "Large Protein Model" trained on billions of evolutionary sequences. ESM-3 has already mastered the grammar of protein biology—it knows which amino acid sequences are stable, valid, and evolutionarily plausible. We don't need to teach it how to "be a peptide"; it already knows.

### 2. The Conditioning: Dealing with "Who?" (Conditioned Generation)

The core innovation is **Target Conditioning**. ESM-3 knows how to generate valid biology, but it doesn't naturally know how to generate a binder for _your_ specific target (e.g., P53 or HER2).

To bridge this gap, we introduce **Target Tokens**:

- We assign a unique token (e.g., `<TARGET_P53>`) to each target protein.
- We train the model on sequences known to bind to that target.
- **The Result**: The model learns a **latent profile** of binding preferences for that specific target. It learns that `<TARGET_P53>` requires a specific hydrophobic motif, while `<TARGET_HER2>` needs a rigid charged loop.

### 3. The Strategy: "Steering the Giant"

We don't train a model from scratch. We use **LoRA (Low-Rank Adaptation)** to slightly adjust the attention mechanisms of ESM-3. This "steers" the massive pre-existing knowledge of the model toward our specific task.

Effectively, we turn a general-purpose "Protein Generator" into a specialized "Peptide Binder Generator" that takes a Target ID as input and "dreams" a compatible binding sequence.

### 4. Implementation Details

We implement this using a **Masked Language Modeling (MLM)** objective, heavily adapted for generation:

1.  **Target Conditioning**:
    - Every sequence is permanently anchored with its target ID: `[TARGET_ID] [BOS] [SEQUENCE...] [EOS]`.
    - This token is never masked, serving as the "prompt" for the generation.

2.  **Training Data vs. Generation Goal**:
    - **Training**: We train on a wide range of known **Binding Sequences** (up to 512 amino acids) to learn valid interaction patterns (the "physics" of binding).
    - **Inference**: We constrain the model to generate short **Peptides** (< 20 amino acids). The model transfers the structural binding principles it learned from the long sequences to create novel, short peptides.

3.  **Aggressive Masking**:
    - We use a dynamic masking ratio of **25% - 75%** (higher than the standard 15%).
    - **Why?** To generate a new peptide, the model must be able to hallucinate valid structures from almost nothing. High masking rates force it to learn global structural dependencies rather than just local sequence repair.
    - **Masking Boost**: A custom loss function (`Boost = 1.0 + 0.5 * log(N_masks + 1)`) creates a curriculum where the model is rewarded exponentially more for solving difficult, heavily masked scenarios.

4.  **Smart Batching**:
    - We use a `LengthGroupedSampler` to group sequences of similar lengths. This creates efficient, dense batches that maximize GPU throughput (essential for training on sequences up to 512aa).

## Directory Structure

```
├── data/                       # Ignored directory for generated datasets
├── mimir/                      # Core package
│   ├── dataset.py              # PyTorch Dataset with dynamic masking logic
│   ├── model_utils.py          # Utilities for resizing ESM-3 embeddings
│   ├── sampler.py              # LengthGroupedSampler for smart batching
│   ├── tokenizer.py            # Wrapper around ESM-3 tokenizer
│   └── __init__.py
├── notebooks/                  # Experimental notebooks
├── scripts/                    # Executable scripts
│   ├── dataset_utils.py        # Shared utilities for dataset generation
│   ├── download_weights.py     # Triggers model weight download
│   ├── estimate_training.py    # Calculates training resource requirements
│   ├── generate_mapping_dataset.py # Generates the Mapping Dataset (Target -> Sequence)
│   ├── generate_peptide_dataset.py # Generates the Peptide Dataset (<20aa)
│   ├── test_esm3.py            # Validates installation
│   └── train.py                # Main training loop with LoRA
├── setup_esm3.sh               # Environment setup script
├── README.md
├── pyproject.toml
└── uv.lock
```

## Setup

This project uses `uv` for dependency management.

1.  **Initialize environment**:

    ```bash
    uv sync
    ```

2.  **Configure Environment**:
    Create a `.env` file (if needing DB access):
    ```
    POSTGRES_HOST=...
    # ... (see .env.example)
    ```

## Usage

### 1. Generate Datasets

We generate two types of datasets:

1.  **Peptide Dataset**: Short sequences (< 20aa) for specific analysis.
2.  **Mapping Dataset**: Comprehensive binding sequences (up to 512aa) for training.

```bash
uv run scripts/generate_peptide_dataset.py
uv run scripts/generate_mapping_dataset.py
```

### 2. Fine-tune ESM-3

Train on the **Mapping Dataset** to learn general binding rules.

```bash
# Set memory fragmentation variables for stable training
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run scripts/train.py \
    --dataset data/mapping_dataset.csv \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --epochs 100 \
    --lr 1e-4 \
    --use_8bit_adam
```

**Key Parameters:**

- `--batch_size`: Per-GPU batch size (e.g., 16 on H100).
- `--gradient_accumulation_steps`: Simulates a larger batch size (e.g., 4 \* 16 = 64 effective batch).
- `--use_8bit_adam`: Saves optimizer memory, allowing larger batches/models.

### 3. Estimate Training Resources

Check how long your training will take based on your specific dataset size and GPU.

```bash
uv run scripts/estimate_training.py
```

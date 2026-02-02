# MÍMIR

**MÍMIR** is a specialized framework for **de novo peptide design**.

By leveraging the generative capabilities of **ESM-3**, MÍMIR outputs novel peptide sequences (< 20 amino acids) conditioned to bind specific human protein targets. It transforms the problem of finding a binder from a random search into a targeted generation task.

## Core Concepts

Finding effective peptide binders for specific protein targets is a challenge of **combinatorial biology**. The number of possible peptide sequences is astronomical (20^sequence_length), making random screening inefficient.

**MÍMIR** solves this by treating biology as a language.

### 1. Foundation: ESM-3

We leverage **ESM-3**, a "Large Protein Model" trained on billions of evolutionary sequences. ESM-3 has already mastered the grammar of protein biology—it knows which amino acid sequences are stable, valid, and evolutionarily plausible. We don't need to teach it how to "be a peptide"; it already knows.

### 2. Conditioning: Target Tokens

The core innovation is **Target Conditioning**. ESM-3 knows how to generate valid biology, but it doesn't naturally know how to generate a binder for _your_ specific target (e.g., P53 or HER2).

To bridge this gap, we introduce **Target Tokens**:

- We assign a unique token (e.g., `<TARGET_P04637>`) to each target protein (identified by UniProt Accession).
- We train the model on sequences known to bind to that target.
- **The Result**: The model learns a **latent profile** of binding preferences for that specific target. It learns that `<TARGET_P04637>` requires a specific hydrophobic motif, while `<TARGET_P04626>` needs a rigid charged loop.

### 3. Adaptation: LoRA

We don't train a model from scratch. We use **LoRA (Low-Rank Adaptation)** to slightly adjust the attention mechanisms of ESM-3. This "steers" the massive pre-existing knowledge of the model toward our specific task.

Effectively, we turn a general-purpose "Protein Generator" into a specialized "Peptide Binder Generator" that takes a Target Token as input and "dreams" a compatible binding sequence.

### 4. Training: Masked Language Modeling

We implement this using a **Masked Language Modeling (MLM)** objective, heavily adapted for generation:

1.  **Target Anchoring**:
    - Every sequence is permanently anchored with its target ID: `[TARGET_ID] [BOS] [SEQUENCE...] [EOS]`.
    - This token is never masked, serving as the "prompt" for the generation.

2.  **Training Data vs. Generation Goal**:
    - **Training**: We train on a wide range of known **Binding Sequences** (up to 512 amino acids) to learn valid interaction patterns (the "physics" of binding).
    - **Inference**: We constrain the model to generate short **Peptides** (< 20 amino acids). The model transfers the structural binding principles it learned from the long sequences to create novel, short peptides.

3.  **Aggressive Masking**:
    - We use a dynamic masking ratio of **25% - 75%** (higher than the standard 15%).
    - **Why?** To generate a new peptide, the model must be able to hallucinate valid structures from almost nothing. High masking rates force it to learn global structural dependencies rather than just local sequence repair.
    - **Masking Boost**: A custom loss function (`Boost = 1.0 + 0.5 * log(N_masks + 1)`) creates a curriculum where the model is rewarded exponentially more for solving difficult, heavily masked scenarios.

### 5. Inference: Parallel Iterative Decoding

Unlike standard language models (like GPT) that generate text left-to-right, MÍMIR uses **Parallel Iterative Decoding** (inspired by MaskGit) to generate peptides. This is more akin to sculpting than writing:

1.  **The Blank Slate**: We start with a fully masked sequence anchored by the target: `[TARGET_ID] [MASK] [MASK] [MASK] [MASK]`.
2.  **Global Vision**: The model looks at the _entire_ blank canvas and predicts possibilities for every position simultaneously.
3.  **Confident Anchoring**: We don't just pick the next word. We pick the **most confident residues** anywhere in the sequence—maybe a critical Arginine at position 3 and a Tryptophan at position 9 that are essential for binding.
4.  **Refinement**: We "lock in" these high-confidence anchors and re-mask the rest. In the next step, the model fills in the gaps _conditioned_ on these anchors.
5.  **Convergence**: We repeat this process, gradually revealing the full peptide.

**Why this matters**: Binding is a spatial, all-or-nothing property. A peptide needs its key interaction points to be practically perfect. This method allows the model to prioritize the critical "binding pharmacophore" first, and then build a supportive scaffold around it.

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

1.  **Initialize Environment**:

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

### 1. Data Generation

We generate two types of datasets:

1.  **Peptide Dataset**: Short sequences (< 20aa) for specific analysis.
2.  **Mapping Dataset**: Comprehensive binding sequences (up to 512aa) for training.

```bash
uv run scripts/generate_peptide_dataset.py
uv run scripts/generate_mapping_dataset.py
```

### 2. Training

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

### 3. Resource Estimation

Check how long your training will take based on your specific dataset size and GPU.

```bash
uv run scripts/estimate_training.py
```

### 4. Generation

Once the model is fine-tuned, you can generate novel peptide binders for your targets using the `scripts/sample_peptides.py` script.

This script applies the **Parallel Iterative Decoding** strategy described in **Core Concepts §5**. It starts with a blank template and iteratively "sculpts" the peptide by unmasking the most confident tokens.

```bash
uv run scripts/sample_peptides.py \
    --checkpoint_path checkpoints/checkpoint-100 \
    --targets "P04637,P04626" \
    --min_size 10 \
    --max_size 20 \
    --num_peptides 5 \
    --temperature 1.0 \
    --top_n 0.25
```

**Key Parameters:**

- `--checkpoint_path`: Path to the directory containing the fine-tuned model checkpoint and `vocab.json`.
- `--targets`: Comma-separated list of target UniProt Accessions (must exist in the training vocabulary).
- `--min_size` / `--max_size`: Length range of peptides to generate (default: 10-20).
- `--num_peptides`: Number of peptides to generate per target and per length (default: 1).
- `--temperature`: Controls sampling randomness (default: 1.0).
- `--top_n`: Controls the generation step size (default: 0.25).
  - If `0 < n < 1` (e.g., `0.25`): Unmasks 25% of the remaining tokens per step.
  - If `n >= 1` (e.g., `1`): Unmasks exactly `n` tokens per step.
- `--device`: Device to run generation on (default: "cuda" if available, else "cpu").

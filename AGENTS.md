# MÍMIR Agent Context

## 1. Project Mission

**MÍMIR** is a generative biology framework designed to "dream" novel peptide binders for specific human proteins. We are not just predicting properties; we are generating **de novo biological matter** using **ESM-3**.

## 2. Technical Mental Model

- **The Engine**: We fine-tune **ESM-3** using **LoRA**. We do not train from scratch.
- **The Paradigm**: Use **Masked Language Modeling**, not Causal LM.
  - _Wrong_: "Predict the next amino acid."
  - _Right_: "Sculpt the sequence from noise (Parallel Iterative Decoding)."
- **The Anchor**: Generation is **Target-Conditioned**. Every sequence starts with a `<TARGET_ID>` token (UniProt Accession), acting as the prompt that steers the model's latent space.

## 3. Operational Guidelines

### Environment

- **Package Manager**: Strict usage of `uv`.
- **Execution**: Always run via `uv run scripts/...`.

### Code & Data

- **Scripts**: We prefer standalone scripts in `scripts/` over complex monolithic package logic.
- **Data flow**:
  - `datasets/` generation -> `data/` (csv)
  - `train.py` -> `checkpoints/`
  - `sample_peptides.py` -> Generation

### Code Style

- **Type Hints**: Mandatory.
- **Docstrings**: Google-style.
- **Simplicity**: Prefer readable, explicit code over clever abstractions.

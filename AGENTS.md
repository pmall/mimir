# Agent Guidelines for Mimir-v2

This document outlines specific rules and guidelines that all agents must follow when working on the Mimir-v2 project.

## 1. Package Management & Execution

- **Use `uv`**: This project uses the `uv` package manager for dependency resolution and script execution.
- **Run with `uv run`**: ALWAYS use `uv run <command>` instead of invoking `python` or `pip` directly.
  - Example: `uv run python scripts/train.py`
  - Example: `uv run pytest`
- **Do not use global python**: Avoid using the system python interpreter directly to ensure reproducibility and environment isolation.

## 2. Code Style

- **Type Hints**: All new functions and classes must include Python type hints.
- **Docstrings**: Include Google-style docstrings for all modules, classes, and functions.

## 3. Training & Models

- **ESM-3**: This project revolves around fine-tuning ESM-3. Ensure all model interactions are compatible with the ESM-3 architecture.
- **Masked Diffusion**: We are using a Masked Diffusion approach (Masked Language Modeling) for generation, not standard Causal LM.

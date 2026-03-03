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

- **Type Hints**: Mandatory on all functions (params + return).
- **Docstrings**: Google-style, on every public function/class.
- **Simplicity**: Prefer readable, explicit code over clever abstractions.

### Logging

- **Module**: Always use Python's `logging` module, never `print()`.
- **Destination**: Log to `sys.stdout` via `logging.StreamHandler(sys.stdout)`.
- **Format**: `"%(asctime)s - %(levelname)s - %(message)s"`.
- **Setup**: Call `logging.basicConfig(...)` inside `main()` after parsing args, never at module level (prevents third-party library noise at import time).
- **Verbose flag**: Pass `level=logging.INFO if args.verbose else logging.WARNING` directly to `basicConfig`. Do not use `logging.disable()`.
- **Levels**: Use `logger.error()` for failures, `logger.warning()` for recoverable issues, `logger.info()` for progress. Never duplicate the same message at two levels.
- **Noisy libraries**: Silence with `logging.getLogger("httpx").setLevel(logging.WARNING)` etc. at module level.

### CLI & Config Design

All scripts follow a consistent pattern for argument handling:

**1. Centralized Config**
All dataset paths are managed through a single `config.json` file located at the root of each run directory (e.g., `data/run78-v2/config.json`). This eliminates repetitive path arguments across scripts.

- Every run directory contains exactly one `config.json` with all dataset paths
- All scripts that consume or produce dataset paths must accept `--config` instead of individual path arguments
- Use `load_config()` from `mimir.config`:
  ```python
  from mimir.config import load_config
  
  config = load_config(args.config)
  # Access paths via config.features_fingerprints, config.binders_merged, etc.
  ```
- **Forbidden**: Individual path arguments like `--fingerprints-lmdb`, `--binders-lmdb`, `--associations-csv` are not allowed
- **Exceptions**: Training outputs (checkpoints, logs) and test fixtures remain CLI args: `--checkpoint-dir`, `--resume-from`, `-o`

**2. Argument Conventions**

- **Naming**: kebab-case (`--min-length`, `--num-workers`)
- **Input/Output**: Always `required=True`, never default paths. Use `-o` shorthand for `--output`
- **Verbose**: `-v` / `--verbose`, `action="store_true"`
- **Defaults**: Document in help string (e.g. `"default: 4"`)

**3. Function Design**

- **Execution pattern**: `main()` contains only argparse, `basicConfig`, input validation, and a call to the business logic function
- **Required params first**: `output: Path` before optional params like `min_len: int = 4`
- **Match CLI names**: CLI `--min-length` → function param `min_len`
- **String encoding**: Use `utf-8` for all `.encode()` calls (LMDB keys, hashing, etc.)

### Async Scripts

- **3-Layer Pattern**: Async scripts must follow a 3-layer pattern: `async def _run(...)` for async logic, a public sync wrapper calling `asyncio.run(_run(...))`, and `main()` for CLI parsing.

### File & Resource Operations

- **LMDB**: Key encoding must be `utf-8`. Define map size as a module-level constant `LMDB_MAP_SIZE`. Read-only opens use `readonly=True, lock=False`.
- **Input validation**: Scripts with file path inputs must validate existence early in `main()` using `logger.error()` + `sys.exit(1)`.
- **Multiprocessing**: Use `spawn` context. Worker globals in `_UPPER_SNAKE`. Set `torch.set_num_threads(1)` in worker init.
- **Warnings**: Suppress third-party warnings at module level with `warnings.filterwarnings()`, never inline inside functions.

### Imports

- **Order**: stdlib → third-party → local (`from mimir...`, `from scripts...`).
- **No inline imports**: All imports at module top level.
- **Style**: Prefer `from X import Y` over `import X as Y`.

### Comments & Structure

- **Section markers**: Use `# ---...--- ` dividers with section names (e.g. `# Constants`, `# Main`).
- **No redundant comments**: Don't comment what the code already says.
- **No old-style type comments**: Use proper type hints, not `# type: dict`.
- **Trailing whitespace**: No trailing blank lines at end of file.

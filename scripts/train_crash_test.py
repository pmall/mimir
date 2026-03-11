"""
Crash test for MÍMIR v2 training — VRAM stress test.

Generates the largest possible batch of synthetic max-length tensors and
runs it through the REAL training loop to test production VRAM limits.

This script uses direct monkeypatching (not unittest.mock.patch) to swap
the dataset class with a synthetic one. No fragile string-based attribute
lookups that break when imports change.

Usage:
    uv run python -m scripts.train_crash_test --batch-size 128 --accum 1

    Lower --batch-size until the test passes without OOM.
    Use that value in your production training command.
"""

# ---------------------------------------------------------------------------
# Stdlib imports
# ---------------------------------------------------------------------------
import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment — must be set BEFORE importing torch
# ---------------------------------------------------------------------------
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Local imports — we import the REAL training module to exercise the actual
# code path. The crash test's value is that it runs the same forward/backward
# pass as production, just with synthetic data.
# ---------------------------------------------------------------------------
import scripts.train as train
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum sequence length in the dataset:
# 1 (BOS) + 280 (max fingerprint) + 1 (CUT) + 96 (max binder) + 1 (EOS) = 379
MAX_LEN = 379

# Enough synthetic samples to fill multiple batches
NUM_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Synthetic Dataset — drop-in replacement for MimirDataset
# ---------------------------------------------------------------------------


class SyntheticDataset(Dataset):
    """Generates synthetic max-length tensors for VRAM stress testing.

    This class has the same interface as MimirDataset:
    - .samples list (needed by BucketBatchSampler for length counting)
    - .fingerprints_lmdb / .binders_lmdb (needed by BucketBatchSampler)
    - __getitem__ returns the same dict structure as MimirDataset

    The synthetic data has correct special token placement so the masking
    logic in train.py works correctly (BOS at 0, chainbreak at 281, EOS at end).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tokenizer = load_tokenizer()
        self.samples = [{"target": "T", "binder": "B"}] * NUM_SAMPLES
        # BucketBatchSampler accesses these paths
        self.fingerprints_lmdb = Path("/dev/null")
        self.binders_lmdb = Path("/dev/null")

    def __len__(self) -> int:
        return NUM_SAMPLES

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Build synthetic tensors mimicking build_input_tensors output."""
        seq = torch.randint(0, 30, (MAX_LEN,), dtype=torch.long)

        # Place special tokens where the masking logic expects them:
        # Position 0: BOS
        # Position 281: chainbreak (after 280 fingerprint tokens + BOS)
        # Position -1: EOS
        seq[0] = self.tokenizer.seq_bos
        seq[281] = self.tokenizer.seq_chainbreak
        seq[-1] = self.tokenizer.seq_eos

        return {
            "sequence": seq,
            "structure": torch.randint(0, 4000, (MAX_LEN,), dtype=torch.long),
            "sasa": torch.randint(0, 15, (MAX_LEN,), dtype=torch.long),
            "sequence_id": torch.ones((MAX_LEN,), dtype=torch.long),
            "chain_id": torch.cat([
                # Chain 1: BOS + fingerprint + chainbreak = 282 tokens
                torch.ones(282, dtype=torch.long),
                # Chain 2: binder + EOS = remaining tokens
                torch.full((MAX_LEN - 282,), 2, dtype=torch.long),
            ]),
            "structure_coords": torch.randn((MAX_LEN, 3, 3), dtype=torch.float32),
            "length": MAX_LEN,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mimir v2 VRAM Crash Test")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to test")
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=== MÍMIR v2 VRAM Crash Test ===")
    logger.info(f"Batch size: {args.batch_size}, Accumulation: {args.accum}")
    logger.info(f"Max sequence length: {MAX_LEN}")

    # Use a temporary directory for checkpoints — auto-cleaned on exit
    with tempfile.TemporaryDirectory(prefix="mimir_crash_") as tmpdir:
        # Build the same argparse.Namespace that train._run() expects.
        # checkpoint_every=9999 means no epoch checkpoints are saved,
        # so we don't need to mock the save functions.
        train_args = argparse.Namespace(
            config=Path("dummy_config.json"),
            checkpoint_dir=tmpdir,
            epochs=1,
            batch_size=args.batch_size,
            peak_lr=1e-4,
            lam=1.0,
            checkpoint_every=9999,
            gradient_accumulation_steps=args.accum,
            use_8bit_adam=True,
            num_workers=0,      # No multiprocessing for crash test
            seed=42,
            verbose=True,
            no_compile=False,   # Still compile — we want to test the real path
            associations_csv="dummy.csv",
            fingerprints_lmdb="dummy_fp",
            binders_lmdb="dummy_bin",
        )

        # --- Monkeypatch ---
        # Direct attribute swap is bulletproof: no string-based lookups,
        # no unittest.mock, no risk of "attribute not found" errors.
        # We restore originals in the finally block.
        original_dataset_cls = train.MimirDataset
        original_scan_lengths = train.BucketBatchSampler._scan_lengths

        train.MimirDataset = SyntheticDataset
        train.BucketBatchSampler._scan_lengths = lambda self: [MAX_LEN] * NUM_SAMPLES

        logger.info("Running full training epoch with synthetic data...")
        try:
            train._run(train_args)
            logger.info("Crash test PASSED — no OOM.")

        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.error("!!! CRASH TEST FAILED: OUT OF MEMORY !!!")
                logger.error(f"Reduce --batch-size below {args.batch_size} and retry.")
            else:
                logger.error(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()
            sys.exit(1)

        finally:
            # Always restore originals to avoid polluting other imports
            train.MimirDataset = original_dataset_cls
            train.BucketBatchSampler._scan_lengths = original_scan_lengths

            # --- VRAM Report ---
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info("--- VRAM REPORT ---")
                logger.info(f"Peak VRAM: {max_mem:.2f} GB / {total_mem:.2f} GB")
                logger.info(f"Utilization: {(max_mem / total_mem) * 100:.1f}%")
                logger.info("-------------------")
            else:
                logger.warning("No CUDA — VRAM report skipped.")


if __name__ == "__main__":
    main()

"""
Crash test script for MÍMIR v2 training.
This script monkeypatches the dataset with synthetic max-length data and runs 
the actual training loop from scripts/train.py to test production VRAM limits.

Usage:
    uv run python -m scripts.train_crash_test --batch-size 128 --accum 1
"""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import os
# Fix segment fragmentation to avoid OOMs on long runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Ensure stdout is not buffered so we easily see logs in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from torch.utils.data import Dataset

# Import the actual training logic
import scripts.train as train
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


# --- Constants ---

MAX_LEN = 379  # 1 (BOS) + 280 (FP) + 1 (CUT) + 96 (BINDER) + 1 (EOS)
NUM_SAMPLES = 1000 # Enough to fill a few batches

# --- Synthetic Dataset ---

class SyntheticDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.tokenizer = load_tokenizer()
        self.samples = [{"target": "T", "binder": "B"}] * NUM_SAMPLES
        # BucketBatchSampler needs .lengths
        self.lengths = [MAX_LEN] * NUM_SAMPLES
        
    def __len__(self):
        return NUM_SAMPLES
        
    def __getitem__(self, idx):
        # Build synthetic tensors exactly as build_input_tensors would
        seq = torch.randint(0, 30, (MAX_LEN,), dtype=torch.long)
        # Set special tokens for masking logic in train.py
        seq[0] = self.tokenizer.seq_bos
        seq[281] = self.tokenizer.seq_chainbreak
        seq[-1] = self.tokenizer.seq_eos
        
        return {
            "sequence": seq,
            "structure": torch.randint(0, 4000, (MAX_LEN,), dtype=torch.long),
            "sasa": torch.randint(0, 15, (MAX_LEN,), dtype=torch.long),
            "sequence_id": torch.ones((MAX_LEN,), dtype=torch.long),
            "chain_id": torch.cat([torch.ones(282, dtype=torch.long), torch.full((MAX_LEN-282,), 2, dtype=torch.long)]),
            "structure_coords": torch.randn((MAX_LEN, 3, 3), dtype=torch.float32),
            "length": MAX_LEN
        }

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Mimir v2 Training Crash Test")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to test")
    parser.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting Crash Test...")
    logger.info(f"Target: Batch Size {args.batch_size}, Accumulation {args.accum}")
    logger.info(f"Using max sequence length: {MAX_LEN}")

    # Prepare mocked arguments for train._run (1 epoch, checkpoint every epoch)
    train_args = argparse.Namespace(
        config=Path("dummy_config.json"),
        checkpoint_dir="crash_test_checkpoints",
        epochs=1,
        batch_size=args.batch_size,
        peak_lr=1e-4,
        lam=1.0,
        checkpoint_every=1,
        gradient_accumulation_steps=args.accum,
        use_8bit_adam=True,
        num_workers=0,
        seed=42,
        verbose=True,
        associations_csv="dummy.csv",
        fingerprints_lmdb="dummy_fp",
        binders_lmdb="dummy_bin",
    )

    with patch("scripts.train.MimirDataset", SyntheticDataset), \
         patch("scripts.train._save_epoch_checkpoint"), \
         patch("scripts.train._save_model_checkpoint"), \
         patch("mimir.dataset.lmdb.open"):
        
        with patch.object(train.BucketBatchSampler, "_scan_lengths", return_value=[MAX_LEN]*NUM_SAMPLES):
            
            logger.info("Executing full training epoch...")
            try:
                train._run(train_args)
                logger.info("Crash test completed successfully (no OOM).")
                
            except Exception as e:
                if "out of memory" in str(e).lower():
                    logger.error("!!! CRASH TEST FAILED: OUT OF MEMORY !!!")
                else:
                    logger.error(f"Unexpected error during crash test: {e}")
                sys.exit(1)
                
            finally:
                if torch.cuda.is_available():
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info("--- VRAM REPORT ---")
                    logger.info(f"Max VRAM Allocated: {max_mem:.2f} GB")
                    logger.info(f"Total Device VRAM: {total_mem:.2f} GB")
                    logger.info(f"Utilization: {(max_mem/total_mem)*100:.1f}%")
                    logger.info("-------------------")
                else:
                    logger.warning("CUDA not available. VRAM report skipped.")

if __name__ == "__main__":
    main()

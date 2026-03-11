"""
Crash test for MÍMIR v2 training — VRAM stress test.

Standalone script that loads the real model, creates synthetic data,
and runs a full forward/backward/optimizer loop. No monkeypatching,
no imports from train.py, no module tricks.

Usage:
    uv run python -m scripts.train_crash_test --batch-size 8 --accum 16
    uv run python -m scripts.train_crash_test --batch-size 4 --accum 32 --no-compile
"""

# ---------------------------------------------------------------------------
# Stdlib imports
# ---------------------------------------------------------------------------
import argparse
import functools
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment — must be set BEFORE importing torch
# ---------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ---------------------------------------------------------------------------
# Local imports — model and tokenizer only, NOT train.py
# ---------------------------------------------------------------------------
from mimir.model import load_model
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum sequence length:
# 1 (BOS) + 280 (max fingerprint) + 1 (CUT) + 96 (max binder) + 1 (EOS) = 379
MAX_LEN = 379
PADDED_LEN = 384  # nearest multiple of 64
NUM_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------


class SyntheticDataset(Dataset):
    """Generates synthetic tensors matching the real MimirDataset output format."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return NUM_SAMPLES

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.randint(0, 30, (MAX_LEN,), dtype=torch.long)

        # Place special tokens exactly where masking expects them
        seq[0] = self.tokenizer.seq_bos
        seq[281] = self.tokenizer.seq_chainbreak
        seq[-1] = self.tokenizer.seq_eos

        # Structure coords: fingerprint has real coords, rest is NaN (matches real data)
        coords = torch.full((MAX_LEN, 3, 3), float('nan'), dtype=torch.float32)
        coords[1:281] = torch.randn((280, 3, 3), dtype=torch.float32)

        return {
            "sequence": seq,
            "structure": torch.randint(0, 4000, (MAX_LEN,), dtype=torch.long),
            "sasa": torch.randint(0, 15, (MAX_LEN,), dtype=torch.long),
            "sequence_id": torch.ones((MAX_LEN,), dtype=torch.long),
            "chain_id": torch.cat([
                torch.ones(282, dtype=torch.long),
                torch.full((MAX_LEN - 282,), 2, dtype=torch.long),
            ]),
            "structure_coords": coords,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stacks and pads batch tensors to PADDED_LEN."""
    batch_size = len(batch)
    tokenizer = load_tokenizer()

    seq = torch.full((batch_size, PADDED_LEN), tokenizer.seq_pad, dtype=torch.long)
    struct = torch.full((batch_size, PADDED_LEN), tokenizer.struct_pad, dtype=torch.long)
    sasa = torch.full((batch_size, PADDED_LEN), tokenizer.sasa_pad, dtype=torch.long)
    seq_id = torch.zeros((batch_size, PADDED_LEN), dtype=torch.long)
    chain = torch.zeros((batch_size, PADDED_LEN), dtype=torch.long)
    coords = torch.full((batch_size, PADDED_LEN, 3, 3), float('nan'), dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["sequence"].size(0)
        seq[i, :L] = item["sequence"]
        struct[i, :L] = item["structure"]
        sasa[i, :L] = item["sasa"]
        seq_id[i, :L] = item["sequence_id"]
        chain[i, :L] = item["chain_id"]
        coords[i, :L] = item["structure_coords"]

    return {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "sequence_id": seq_id,
        "chain_id": chain,
        "structure_coords": coords,
    }


# ---------------------------------------------------------------------------
# Masking (copied from train.py — no imports needed)
# ---------------------------------------------------------------------------


def apply_masking(batch: dict, tokenizer: Any) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Masks the binder region with independent sequence/structure rates."""
    seq = batch["sequence"].clone()
    struct = batch["structure"].clone()
    labels_seq = torch.full_like(seq, -100)
    labels_struct = torch.full_like(struct, -100)

    for i in range(seq.size(0)):
        cut_pos_t = (seq[i] == tokenizer.seq_chainbreak).nonzero(as_tuple=True)[0]
        if len(cut_pos_t) == 0:
            continue
        cut_pos = cut_pos_t[0].item()

        eos_pos_t = (seq[i] == tokenizer.seq_eos).nonzero(as_tuple=True)[0]
        eos_pos = eos_pos_t[0].item() if len(eos_pos_t) > 0 else seq.size(1)

        binder_start = cut_pos + 1
        binder_end = eos_pos
        binder_len = binder_end - binder_start
        if binder_len <= 0:
            continue

        # Sequence masking
        rate = random.uniform(0.25, 0.75)
        n_mask = max(1, int(round(binder_len * rate)))
        indices = random.sample(range(binder_start, binder_end), n_mask)
        for idx in indices:
            labels_seq[i, idx] = seq[i, idx].item()
            seq[i, idx] = tokenizer.seq_mask

        # Structure masking (Case A only)
        struct_binder = struct[i, binder_start:binder_end]
        if not torch.all(struct_binder == tokenizer.struct_mask):
            rate_s = random.uniform(0.25, 0.75)
            n_mask_s = max(1, int(round(binder_len * rate_s)))
            indices_s = random.sample(range(binder_start, binder_end), n_mask_s)
            for idx in indices_s:
                labels_struct[i, idx] = struct[i, idx].item()
                struct[i, idx] = tokenizer.struct_mask

    masked = {
        "sequence": seq, "structure": struct, "sasa": batch["sasa"],
        "chain_id": batch["chain_id"], "structure_coords": batch["structure_coords"],
        "sequence_id": batch["sequence_id"],
    }
    return masked, labels_seq, labels_struct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MÍMIR v2 VRAM Crash Test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=== MÍMIR v2 VRAM Crash Test ===")
    logger.info(f"Batch size: {args.batch_size}, Accumulation: {args.accum}")
    logger.info(f"Max sequence length: {MAX_LEN} (padded to {PADDED_LEN})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- GPU optimizations (same as train.py) ---
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("Flash Attention / SDPA enabled (with math fallback).")

    # --- Model ---
    logger.info("Loading model...")
    model = load_model()
    model.to(device)
    model.train()

    if not args.no_compile and torch.cuda.is_available():
        logger.info("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)
    else:
        logger.info("Skipping torch.compile.")

    # --- Optimizer ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(trainable, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    # --- Data ---
    tokenizer = load_tokenizer()
    dataset = SyntheticDataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Autocast dtype ---
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        autocast_device = "cuda"
    else:
        autocast_dtype = torch.bfloat16
        autocast_device = "cpu"
    logger.info(f"Autocast: {autocast_device}, {autocast_dtype}")

    # --- Loss function ---
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    # --- Training loop ---
    logger.info(f"Running {len(dataloader)} batches (accum={args.accum})...")
    optimizer.zero_grad()
    num_batches = 0

    try:
        pbar = tqdm(dataloader, desc="Crash Test")
        for step, batch in enumerate(pbar):
            masked, labels_seq, labels_struct = apply_masking(batch, tokenizer)

            tokens = {k: v.to(device) for k, v in masked.items()}
            labels_seq = labels_seq.to(device)
            labels_struct = labels_struct.to(device)

            with torch.amp.autocast(autocast_device, dtype=autocast_dtype):
                output = model(
                    sequence_tokens=tokens["sequence"],
                    structure_tokens=tokens["structure"],
                    sasa_tokens=tokens["sasa"],
                    chain_id=tokens["chain_id"],
                    structure_coords=tokens["structure_coords"],
                    sequence_id=tokens["sequence_id"],
                )

                loss_seq = criterion(
                    output.sequence_logits.float().view(-1, output.sequence_logits.size(-1)),
                    labels_seq.view(-1),
                ).view(labels_seq.size())

                loss_struct = criterion(
                    output.structure_logits.float().view(-1, output.structure_logits.size(-1)),
                    labels_struct.view(-1),
                ).view(labels_struct.size())

                mask_seq = labels_seq != -100
                mask_struct = (labels_struct != -100) & (labels_struct != tokenizer.struct_nan)
                total_masked = mask_seq.sum() + mask_struct.sum()

                if total_masked > 0:
                    loss = ((loss_seq * mask_seq.float()).sum() + (loss_struct * mask_struct.float()).sum()) / total_masked.float()
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                loss = loss / args.accum

            # Backward
            loss.backward()

            num_batches += 1
            if num_batches % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(Loss=f"{loss.item() * args.accum:.4f}")

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
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("--- VRAM REPORT ---")
            logger.info(f"Peak VRAM: {peak:.2f} GB / {total:.2f} GB")
            logger.info(f"Utilization: {(peak / total) * 100:.1f}%")
            logger.info("-------------------")


if __name__ == "__main__":
    main()

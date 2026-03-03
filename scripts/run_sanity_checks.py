"""
Pre-Training Sanity Checks for Mimir v2.

Runs lightweight CPU checks to verify data integrity, dataloader correctness,
and a simple mock smoke test without loading the real ESM3 model.

Usage:
    uv run python -m scripts.run_sanity_checks \\
        --config data/run78-v2/config.json [-v]
"""

import os
import sys
import argparse
import logging
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import lmdb
import msgpack
import pandas as pd

from mimir.config import load_config
from mimir.tokenizer import load_tokenizer, MimirTokenizer, CUT_TOKEN_ID_SEQ, CUT_TOKEN_ID_STRUCT
from mimir.dataset import MimirDataset, BucketBatchSampler, mimir_collate_fn
from scripts.train import apply_mlm_masking
from tests.mocks import MockEsm3

# --- Noisy library silencing (module level) ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmdb").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# --- Constants ---
MAX_FP_LEN = 280
MAX_BINDER_LEN = 96
LMDB_MAP_SIZE = 100 * 1024 * 1024 * 1024  # 100 GB
SEQ_VOCAB_SIZE = CUT_TOKEN_ID_SEQ + 1      # 65
STRUCT_VOCAB_SIZE = CUT_TOKEN_ID_STRUCT + 1  # 4101
MOCK_HIDDEN = 16


# --- Helpers ---

def _deserialize(value: bytes) -> dict:
    """Unpacks a msgpack-encoded bytes value."""
    return msgpack.unpackb(value, raw=False)


# --- Checks ---

def check_environment() -> None:
    logger.info("--- 1. Environment ---")
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if HAS_BNB:
        logger.info("bitsandbytes: OK")
    else:
        logger.warning("bitsandbytes: not importable")


def check_data_integrity(
    fingerprints_lmdb: Path,
    binders_lmdb: Path,
    associations_csv: Path,
) -> None:
    logger.info("--- 2. Data Integrity ---")
    fp_env = lmdb.open(str(fingerprints_lmdb), readonly=True, lock=False)
    bd_env = lmdb.open(str(binders_lmdb), readonly=True, lock=False)

    with fp_env.begin() as txn:
        logger.info(f"Fingerprints LMDB entries: {txn.stat()['entries']}")
    with bd_env.begin() as txn:
        logger.info(f"Binders LMDB entries: {txn.stat()['entries']}")

    df = pd.read_csv(associations_csv)
    logger.info(f"CSV rows: {len(df)}")

    target_col = next((c for c in ("uniprot_accession", "target_id", "target") if c in df.columns), None)
    binder_col = next((c for c in ("binder_id", "binder") if c in df.columns), None)
    assert target_col and binder_col, "Could not find target/binder columns in CSV"
    assert df[[target_col, binder_col]].isnull().sum().sum() == 0, "Null values in key columns"

    found_fp, found_bd, has_struct = 0, 0, 0
    sample = df.sample(min(1000, len(df)), random_state=42)
    with fp_env.begin() as ftxn, bd_env.begin() as btxn:
        for _, row in sample.iterrows():
            fp = ftxn.get(str(row[target_col]).encode("utf-8"))
            bd = btxn.get(str(row[binder_col]).encode("utf-8"))
            if fp:
                found_fp += 1
            if bd:
                found_bd += 1
                if _deserialize(bd).get("structure") is not None:
                    has_struct += 1
    logger.info(f"Fingerprints found: {found_fp}/{len(sample)}")
    logger.info(f"Binders found: {found_bd}/{len(sample)}")
    logger.info(f"With structure: {has_struct}/{found_bd}")
    if found_fp < len(sample) * 0.95 or found_bd < len(sample) * 0.95:
        logger.warning(">5% of cross-reference lookups failed")

    fp_lengths = []
    with fp_env.begin() as txn:
        for i, (_, v) in enumerate(txn.cursor()):
            if i >= 500:
                break
            fp_lengths.append(len(_deserialize(v).get("position_ids", [])))
    if fp_lengths:
        logger.info(f"FP lengths — min:{min(fp_lengths)} max:{max(fp_lengths)} mean:{statistics.mean(fp_lengths):.1f}")
        assert max(fp_lengths) <= MAX_FP_LEN, f"Fingerprint exceeds {MAX_FP_LEN} token limit"

    with bd_env.begin() as txn:
        for i, (_, v) in enumerate(txn.cursor()):
            if i >= 500:
                break
            bd = _deserialize(v)
            assert len(bd.get("sequence", "")) <= MAX_BINDER_LEN, "Binder too long"
            struct = bd.get("structure")
            assert struct is None or isinstance(struct, list), "Structure must be None or list"
            assert struct != [], "Structure cannot be empty list"

    fp_env.close()
    bd_env.close()
    logger.info("Data Integrity: PASS")


def check_tokenizer() -> MimirTokenizer:
    logger.info("--- 3. Tokenizer ---")
    tokenizer = load_tokenizer()
    cut_id = tokenizer.cut_seq
    assert cut_id is not None, "<cut> not registered"
    logger.info(f"<cut> token ID: {cut_id}")
    logger.info("Tokenizer: PASS")
    return tokenizer


def check_dataloader(
    fingerprints_lmdb: Path,
    binders_lmdb: Path,
    associations_csv: Path,
    tokenizer: MimirTokenizer,
) -> torch.utils.data.DataLoader:
    logger.info("--- 4. Dataloader ---")
    dataset = MimirDataset(
        associations_csv=associations_csv,
        fingerprints_lmdb=fingerprints_lmdb,
        binders_lmdb=binders_lmdb,
        tokenizer=tokenizer,
    )
    sampler = BucketBatchSampler(dataset, batch_size=4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: mimir_collate_fn(b, tokenizer),
        num_workers=0,
    )
    lengths, pad_counts = [], []
    it = iter(dataloader)
    for _ in range(min(5, len(sampler))):
        try:
            batch = next(it)
        except StopIteration:
            break
        if not batch:
            continue
        L = batch["attention_mask"].sum(dim=1)
        lengths.extend(L.tolist())
        pad_counts.append(batch["sequence"].shape[1] * len(L) - L.sum().item())
    if lengths:
        pad_frac = sum(pad_counts) / (sum(lengths) + sum(pad_counts))
        logger.info(f"Mean length: {sum(lengths)/len(lengths):.1f}, pad fraction: {pad_frac:.3f}")
        if pad_frac > 0.15:
            logger.warning("Pad fraction >15%")
    logger.info("Dataloader: PASS")
    return dataloader


def run_smoke_test(
    dataloader: torch.utils.data.DataLoader,
    tokenizer: MimirTokenizer,
) -> None:
    logger.info("--- 5. Smoke Test ---")
    model = MockEsm3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(dataloader)
    for epoch in range(2):
        for step in range(2):
            try:
                batch = next(it)
            except StopIteration:
                break
            if not batch:
                continue
            optimizer.zero_grad()
            masked_batch, labels_seq, _ = apply_mlm_masking(batch, tokenizer)
            out = model(
                sequence_tokens=masked_batch["sequence"],
                structure_tokens=masked_batch["structure"],
                sasa_tokens=masked_batch["sasa"],
                position_ids=masked_batch["position_ids"],
            )
            # Dummy loss using sum of logits for simplistic mock backward pass
            loss = out.sequence_logits.sum() + out.structure_logits.sum()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            logger.info(f"epoch {epoch} step {step} dummy loss computed")
    logger.info("Smoke Test: PASS")


# --- Orchestration ---

def run_checks(
    fingerprints_lmdb: Path,
    binders_lmdb: Path,
    associations_csv: Path,
) -> None:
    check_environment()
    check_data_integrity(fingerprints_lmdb, binders_lmdb, associations_csv)
    tokenizer = check_tokenizer()
    dataloader = check_dataloader(fingerprints_lmdb, binders_lmdb, associations_csv, tokenizer)
    run_smoke_test(dataloader, tokenizer)
    logger.info("=== ALL PRE-TRAINING SANITY CHECKS PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training sanity checks for Mimir v2")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    config = load_config(args.config)

    for p in (config.features_fingerprints, config.features_binders, config.binders_merged):
        if not p.exists():
            logger.error(f"Not found: {p}")
            sys.exit(1)

    run_checks(
        fingerprints_lmdb=config.features_fingerprints,
        binders_lmdb=config.features_binders,
        associations_csv=config.binders_merged,
    )


if __name__ == "__main__":
    main()

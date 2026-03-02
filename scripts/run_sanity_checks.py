"""
Pre-Training Sanity Checks for Mimir v2 (Task 4)
Runs lightweight CPU checks to verify data integrity, dataloader correctness,
and training loop structure without loading the real ESM3 model.
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
from transformers import get_cosine_schedule_with_warmup

from mimir.tokenizer import load_tokenizer, MimirTokenizer, CUT_TOKEN_ID_SEQ
from mimir.dataset import MimirDataset, BucketBatchSampler, mimir_collate_fn
from scripts.train import apply_mlm_masking

# --- Noisy library silencing (module level) ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmdb").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Constants ---
MAX_FP_LEN = 280
MAX_BINDER_LEN = 96
LMDB_MAP_SIZE = 100 * 1024 * 1024 * 1024  # 100 GB
# Sequence vocab: IDs 0-63 (standard) + ID 64 (<cut>) => size 65
SEQ_VOCAB_SIZE = CUT_TOKEN_ID_SEQ + 1  # 65
STRUCT_VOCAB_SIZE = 4101               # IDs 0-4096 standard + 4100 <cut>
MOCK_HIDDEN = 16


# --- Mock model ---

class _MockOutput:
    """Holds logits returned by MockEsm3, mimicking ESM3 output attribute access."""

    def __init__(self, sequence_logits: torch.Tensor, structure_logits: torch.Tensor) -> None:
        self.sequence_logits = sequence_logits
        self.structure_logits = structure_logits


class MockEsm3(nn.Module):
    """
    Tiny model that has the same call signature as ESM3 and returns
    logits of the correct shape without loading ESM3.
    """

    def __init__(
        self,
        vocab_seq: int = SEQ_VOCAB_SIZE,
        vocab_struct: int = STRUCT_VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.seq_embed = nn.Embedding(vocab_seq, MOCK_HIDDEN)
        self.struct_embed = nn.Embedding(vocab_struct, MOCK_HIDDEN)
        self.seq_head = nn.Linear(MOCK_HIDDEN, vocab_seq)
        self.struct_head = nn.Linear(MOCK_HIDDEN, vocab_struct)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> _MockOutput:
        """Returns random-init logits of shape (B, L, vocab_size)."""
        hidden = (
            self.seq_embed(sequence_tokens.clamp(0, SEQ_VOCAB_SIZE - 1))
            + self.struct_embed(structure_tokens.clamp(0, STRUCT_VOCAB_SIZE - 1))
        )
        return _MockOutput(self.seq_head(hidden), self.struct_head(hidden))


# --- Helpers ---

def _masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss that returns 0.0 (not nan) when all labels are -100.

    Args:
        logits: Shape (B, L, V).
        labels: Shape (B, L), with -100 for ignored positions.

    Returns:
        Scalar loss tensor.
    """
    valid = labels != -100
    if not valid.any():
        return torch.tensor(0.0, requires_grad=False)
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def _mock_loss(output: _MockOutput, labels_seq: torch.Tensor) -> torch.Tensor:
    """
    Combined seq + struct loss so that every mock parameter receives a gradient.

    Struct labels are derived from seq labels clamped to the struct vocab range.
    """
    labels_struct = labels_seq.clamp(max=STRUCT_VOCAB_SIZE - 1)
    return _masked_ce_loss(output.sequence_logits, labels_seq) + \
           _masked_ce_loss(output.structure_logits, labels_struct)


def _get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Returns a cosine schedule with linear warmup."""
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, step: int) -> None:
    """Saves model weights and training state to path."""
    torch.save({"epoch": epoch, "step": step, "model": model.state_dict()}, path / "ckpt.pt")


def _load_checkpoint(path: Path, model: nn.Module) -> tuple[int, int]:
    """
    Loads model weights from checkpoint if one exists.

    Returns:
        (epoch, step) or (0, 0) if no checkpoint found.
    """
    ckpt = path / "ckpt.pt"
    if not ckpt.exists():
        return 0, 0
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    return state["epoch"], state["step"]


def _deserialize(value: bytes) -> dict:
    """Unpacks a msgpack-encoded bytes value."""
    return msgpack.unpackb(value, raw=False)


# --- Checks ---

def check_environment() -> None:
    """Checks PyTorch version and that key packages are importable."""
    logger.info("--- 1. Environment ---")
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    try:
        import bitsandbytes  # noqa: F401
        logger.info("bitsandbytes: OK")
    except ImportError:
        logger.warning("bitsandbytes: not importable")




def check_data_integrity(
    fingerprints_lmdb: Path,
    binders_lmdb: Path,
    associations_csv: Path,
) -> None:
    """
    Verifies LMDB counts, CSV integrity, cross-reference coverage,
    and sequence/structure length distributions.

    Args:
        fingerprints_lmdb: Path to fingerprints LMDB.
        binders_lmdb: Path to binders LMDB.
        associations_csv: Path to associations CSV.
    """
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
    """
    Verifies the <cut> token is registered and does not collide with other tokens.

    Returns:
        The loaded MimirTokenizer instance.
    """
    logger.info("--- 3. Tokenizer ---")
    tokenizer = load_tokenizer()
    cut_id = tokenizer.cut_seq
    assert cut_id is not None, "<cut> not registered"
    assert cut_id not in (tokenizer.seq_bos, tokenizer.seq_eos, tokenizer.seq_mask, tokenizer.seq_pad)
    standard_ids = {tokenizer.sequence.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    assert cut_id not in standard_ids, "<cut> collides with amino acid token"
    logger.info(f"<cut> token ID: {cut_id}")
    logger.info("Tokenizer: PASS")
    return tokenizer


def check_dataloader(
    fingerprints_lmdb: Path,
    binders_lmdb: Path,
    associations_csv: Path,
    tokenizer: MimirTokenizer,
) -> torch.utils.data.DataLoader:
    """
    Builds a DataLoader and checks bucket-pad distribution over a few batches.

    Args:
        fingerprints_lmdb: Path to fingerprints LMDB.
        binders_lmdb: Path to binders LMDB.
        associations_csv: Path to associations CSV.
        tokenizer: MimirTokenizer instance.

    Returns:
        Configured DataLoader.
    """
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


def check_model_and_loss(
    dataloader: torch.utils.data.DataLoader,
    tokenizer: MimirTokenizer,
) -> MockEsm3:
    """
    Forward pass shape check, loss + backward check, zero-mask check, and LR schedule check.

    Args:
        dataloader: DataLoader producing batches.
        tokenizer: MimirTokenizer instance.

    Returns:
        Mock model after backward pass.
    """
    logger.info("--- 5. Model & Loss (mock, CPU) ---")
    model = MockEsm3()

    batch = next(iter(dataloader))
    if not batch:
        logger.error("Empty batch from dataloader")
        sys.exit(1)

    masked_batch, labels_seq, _labels_struct = apply_mlm_masking(batch, tokenizer)

    # Shape check (no grad needed)
    with torch.no_grad():
        out_check = model(
            sequence_tokens=masked_batch["sequence"],
            structure_tokens=masked_batch["structure"],
            sasa_tokens=masked_batch["sasa"],
            position_ids=masked_batch["position_ids"],
        )
    logger.info(f"seq_logits shape: {out_check.sequence_logits.shape}")
    logger.info(f"struct_logits shape: {out_check.structure_logits.shape}")

    # Loss + backward (fresh forward with grad)
    out = model(
        sequence_tokens=masked_batch["sequence"],
        structure_tokens=masked_batch["structure"],
        sasa_tokens=masked_batch["sasa"],
        position_ids=masked_batch["position_ids"],
    )
    loss = _mock_loss(out, labels_seq)
    assert not torch.isnan(loss) and not torch.isinf(loss), f"Loss is {loss.item()}"
    assert loss.item() > 0, "Loss is 0 — no masked tokens in batch?"
    logger.info(f"Mock loss: {loss.item():.4f}")

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    logger.info("All gradients present and finite.")

    # Zero-mask check: all-(-100) labels => _masked_ce_loss returns 0.0
    zero_labels = torch.full_like(labels_seq, -100)
    out_zero = model(
        sequence_tokens=batch["sequence"],
        structure_tokens=batch["structure"],
        sasa_tokens=batch["sasa"],
        position_ids=batch["position_ids"],
    )
    loss_zero = _mock_loss(out_zero, zero_labels)
    assert loss_zero.item() == 0.0, f"Expected 0.0 for zero-mask, got {loss_zero.item()}"
    logger.info("Zero-mask loss: 0.0 OK")

    # LR schedule check
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps, warmup_steps = 1000, 50
    scheduler = _get_lr_scheduler(optimizer, warmup_steps, total_steps)
    lrs: dict[int, float] = {}
    for step in range(total_steps):
        if step in (0, 25, 50, 500, 999):
            lrs[step] = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
    logger.info(f"LR schedule sample: { {k: f'{v:.2e}' for k, v in lrs.items()} }")
    assert lrs[0] < lrs[50], "LR should increase during warmup"
    assert lrs[50] > lrs[999], "LR should decrease after warmup"

    logger.info("Model & Loss: PASS")
    return model


def check_checkpoint(model: MockEsm3, checkpoint_dir: Path) -> None:
    """
    Verifies checkpoint save/reload correctness.

    Args:
        model: Mock model with known weights.
        checkpoint_dir: Directory for the test checkpoint.
    """
    logger.info("--- 6. Checkpoint ---")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _save_checkpoint(checkpoint_dir, model, epoch=5, step=10)

    model2 = MockEsm3()
    epoch2, step2 = _load_checkpoint(checkpoint_dir, model2)
    assert epoch2 == 5 and step2 == 10

    for (n1, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"

    missing = Path("/tmp/_mimir_sanity_missing")
    ep, sp = _load_checkpoint(missing, model2)
    assert ep == 0 and sp == 0
    logger.info("Checkpoint: PASS")


def run_smoke_test(
    dataloader: torch.utils.data.DataLoader,
    model: MockEsm3,
    tokenizer: MimirTokenizer,
) -> None:
    """
    Runs 2 mini-epochs (2 steps each) of the mock training loop.

    Args:
        dataloader: DataLoader to iterate.
        model: Mock model.
        tokenizer: MimirTokenizer instance.
    """
    logger.info("--- 7. Smoke Test ---")
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
            loss = _mock_loss(out, labels_seq)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            logger.info(f"epoch {epoch} step {step} loss {loss.item():.4f}")
    logger.info("Smoke Test: PASS")


# --- Orchestration ---

def run_checks(
    fingerprints_lmdb: Path,
    binders_lmdb: Path,
    associations_csv: Path,
    checkpoint_dir: Path,
) -> None:
    """
    Runs all pre-training sanity checks in order.

    Args:
        fingerprints_lmdb: Path to fingerprints LMDB.
        binders_lmdb: Path to binders LMDB.
        associations_csv: Path to associations CSV.
        checkpoint_dir: Temporary directory for checkpoint test.
    """
    check_environment()
    check_data_integrity(fingerprints_lmdb, binders_lmdb, associations_csv)
    tokenizer = check_tokenizer()
    dataloader = check_dataloader(fingerprints_lmdb, binders_lmdb, associations_csv, tokenizer)
    model = check_model_and_loss(dataloader, tokenizer)
    check_checkpoint(model, checkpoint_dir)
    run_smoke_test(dataloader, model, tokenizer)
    logger.info("=== ALL PRE-TRAINING SANITY CHECKS PASSED ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training sanity checks for Mimir v2")
    parser.add_argument("--fingerprints-lmdb", type=Path, required=True)
    parser.add_argument("--binders-lmdb", type=Path, required=True)
    parser.add_argument("--associations-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    for p in (args.fingerprints_lmdb, args.binders_lmdb, args.associations_csv):
        if not p.exists():
            logger.error(f"Not found: {p}")
            sys.exit(1)

    run_checks(
        fingerprints_lmdb=args.fingerprints_lmdb,
        binders_lmdb=args.binders_lmdb,
        associations_csv=args.associations_csv,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

"""
Tests for Mimir v2 Training Logic — Unit tests for extracted functions.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from mimir.tokenizer import load_tokenizer, build_input_tensors
from mimir.dataset import mimir_collate_fn
from scripts.train import (
    apply_mlm_masking,
    compute_mlm_loss,
    _compute_detailed_metrics,
    safe_div,
    _compute_epoch_metrics,
    _build_log_entry,
    _build_scheduler,
    _resolve_resume_state,
)


# --- Fixtures ---


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()


@pytest.fixture
def struct_fix():
    with open("tests/data/struct_0.json", "r") as f:
        return json.load(f)


@pytest.fixture
def no_struct_fix():
    with open("tests/data/no_struct_0.json", "r") as f:
        return json.load(f)


# --- safe_div ---


def test_safe_div_normal():
    """Test: safe_div returns correct result for non-zero denominator."""
    assert safe_div(10.0, 2.0) == 5.0


def test_safe_div_zero_denominator():
    """Test: safe_div returns 0.0 when denominator is zero."""
    assert safe_div(10.0, 0.0) == 0.0


def test_safe_div_zero_numerator():
    """Test: safe_div returns 0.0 when numerator is zero."""
    assert safe_div(0.0, 5.0) == 0.0


# --- _compute_epoch_metrics ---


def test_compute_epoch_metrics_basic():
    """Test: _compute_epoch_metrics derives correct accuracy, loss, and perplexity."""
    m = {
        "overall_correct": 7, "overall_total": 10, "overall_loss": 20.0,
        "full_seq_correct": 3, "full_seq_total": 5, "full_seq_loss": 10.0,
        "full_struct_correct": 2, "full_struct_total": 4, "full_struct_loss": 8.0,
        "partial_seq_correct": 2, "partial_seq_total": 1, "partial_seq_loss": 2.0,
    }

    em = _compute_epoch_metrics(m)

    # Overall
    assert em["overall_acc"] == pytest.approx(0.7)
    assert em["overall_loss_raw"] == pytest.approx(2.0)
    assert em["overall_ppl"] == pytest.approx(math.exp(2.0))

    # Full seq
    assert em["full_seq_acc"] == pytest.approx(0.6)
    assert em["full_seq_loss_raw"] == pytest.approx(2.0)

    # Full struct
    assert em["full_struct_acc"] == pytest.approx(0.5)
    assert em["full_struct_loss_raw"] == pytest.approx(2.0)

    # Full combined (seq + struct)
    assert em["full_acc"] == pytest.approx(5 / 9)
    assert em["full_loss_raw"] == pytest.approx(18.0 / 9)

    # Partial seq
    assert em["partial_seq_acc"] == pytest.approx(2.0)
    assert em["partial_seq_loss_raw"] == pytest.approx(2.0)


def test_compute_epoch_metrics_all_zeros():
    """Test: _compute_epoch_metrics handles all-zero counters without division errors."""
    m = {
        "overall_correct": 0, "overall_total": 0, "overall_loss": 0.0,
        "full_seq_correct": 0, "full_seq_total": 0, "full_seq_loss": 0.0,
        "full_struct_correct": 0, "full_struct_total": 0, "full_struct_loss": 0.0,
        "partial_seq_correct": 0, "partial_seq_total": 0, "partial_seq_loss": 0.0,
    }

    em = _compute_epoch_metrics(m)

    assert em["overall_acc"] == 0.0
    assert em["overall_ppl"] == 0.0
    assert em["full_ppl"] == 0.0
    assert em["partial_seq_ppl"] == 0.0


# --- _build_log_entry ---


def test_build_log_entry_contains_all_fields():
    """Test: _build_log_entry produces a dict with all expected keys."""
    em = {
        "overall_acc": 0.7, "overall_loss_raw": 2.0, "overall_ppl": 7.39,
        "full_acc": 0.6, "full_loss_raw": 1.8, "full_ppl": 6.05,
        "full_seq_acc": 0.65, "full_seq_loss_raw": 1.5, "full_seq_ppl": 4.48,
        "full_struct_acc": 0.55, "full_struct_loss_raw": 2.1, "full_struct_ppl": 8.17,
        "partial_seq_acc": 0.8, "partial_seq_loss_raw": 1.2, "partial_seq_ppl": 3.32,
    }

    entry = _build_log_entry(
        epoch=5, avg_loss=3.14, current_lr=5e-5, lam=1.0, total_skipped=3, em=em,
    )

    assert entry["epoch"] == 5
    assert entry["loss"] == 3.14
    assert entry["lr"] == 5e-5
    assert entry["lambda"] == 1.0
    assert entry["skipped_samples"] == 3

    assert entry["overall_accuracy"] == 0.7
    assert entry["overall_perplexity"] == 7.39
    assert entry["overall_loss_raw"] == 2.0

    assert entry["full_accuracy"] == 0.6
    assert entry["full_seq_accuracy"] == 0.65
    assert entry["full_struct_accuracy"] == 0.55
    assert entry["partial_seq_accuracy"] == 0.8

    # Verify it serializes to valid JSON
    json.dumps(entry)


# --- _build_scheduler ---


def test_build_scheduler_warmup_then_decay():
    """Test: scheduler ramps up during warmup and decays to min LR."""
    peak_lr = 1e-4
    total_steps = 100
    dummy = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([dummy], lr=peak_lr)

    scheduler = _build_scheduler(optimizer, total_steps, peak_lr)

    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    warmup_end = int(0.05 * total_steps)

    # Warmup increases
    assert lrs[0] < lrs[warmup_end]

    # Decay decreases
    assert lrs[warmup_end] > lrs[-1]

    # Final LR is approximately 1e-5
    assert math.isclose(scheduler.get_last_lr()[0], 1e-5, rel_tol=1e-3)


# --- _resolve_resume_state ---


def test_resolve_resume_state_empty_dir():
    """Test: _resolve_resume_state returns defaults when no log file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        start_epoch, ckpt_path, state, best_loss = _resolve_resume_state(Path(tmpdir))

        assert start_epoch == 0
        assert ckpt_path is None
        assert state is None
        assert best_loss == float("inf")


def test_resolve_resume_state_with_checkpoint():
    """Test: _resolve_resume_state correctly reads the last checkpoint from the log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a checkpoint directory with training_state.pt
        ckpt_dir = tmpdir / "epoch_3"
        ckpt_dir.mkdir()
        torch.save({"epoch": 3, "optimizer": {}, "scheduler": {}}, ckpt_dir / "training_state.pt")

        # Write a log file
        log_file = tmpdir / "training_log.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({"epoch": 1, "overall_loss_raw": 5.0}) + "\n")
            f.write(json.dumps({"epoch": 2, "overall_loss_raw": 3.0}) + "\n")
            f.write(json.dumps({"epoch": 3, "overall_loss_raw": 4.0}) + "\n")

        start_epoch, ckpt_path, state, best_loss = _resolve_resume_state(tmpdir)

        assert start_epoch == 3
        assert ckpt_path == str(ckpt_dir)
        assert state is not None
        assert state["epoch"] == 3
        # Best loss should be 3.0 (epoch 2), not 4.0 (last epoch)
        assert best_loss == 3.0


def test_resolve_resume_state_missing_checkpoint_exits(tmp_path):
    """Test: _resolve_resume_state exits when log references a missing checkpoint."""
    log_file = tmp_path / "training_log.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"epoch": 5, "overall_loss_raw": 2.0}) + "\n")

    with pytest.raises(SystemExit):
        _resolve_resume_state(tmp_path)


# --- MLM Masking ---


def test_mlm_masking_rate_and_independence(tokenizer, struct_fix):
    """Test: MLM masking rate in [0.25, 0.75], and independent per track at sample level."""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, attn, chain_id, coords = build_input_tensors(fp, binder, tokenizer)

    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "chain_id": chain_id,
        "structure_coords": coords,
        "sequence_id": attn,
        "length": len(seq),
    }

    B = 1000
    batch = mimir_collate_fn([item] * B, tokenizer)

    fp_len = len(fp["sequence"])
    eos_pos = (batch["sequence"][0] == tokenizer.seq_eos).nonzero(as_tuple=True)[0][0].item()
    binder_start = 1 + fp_len + 1
    binder_end = eos_pos
    binder_slice = slice(binder_start, binder_end)
    binder_len = binder_end - binder_start

    masked, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)

    # Rates within range
    seq_rates = (labels_seq[:, binder_slice] != -100).float().sum(dim=1) / binder_len
    struct_rates = (labels_struct[:, binder_slice] != -100).float().sum(dim=1) / binder_len

    assert torch.all(seq_rates >= 0.24) and torch.all(seq_rates <= 0.76)
    assert torch.all(struct_rates >= 0.24) and torch.all(struct_rates <= 0.76)

    # Independence: rates should differ across the batch
    rate_diffs = (seq_rates - struct_rates).abs()
    assert rate_diffs.max() > 0.05, "Masking rates appear coupled"

    # Mask position independence
    same_mask = (labels_seq[:, binder_slice] != -100) == (labels_struct[:, binder_slice] != -100)
    assert same_mask.float().mean().item() < 0.90


def test_mlm_masking_no_struct(tokenizer, no_struct_fix):
    """Test: Case B (no structure) — structure track is not masked."""
    fp, binder = no_struct_fix["fingerprint"], no_struct_fix["binder"]
    seq, struct, sasa, attn, chain_id, coords = build_input_tensors(fp, binder, tokenizer)

    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "chain_id": chain_id,
        "structure_coords": coords,
        "sequence_id": attn,
        "length": len(seq),
    }

    batch = mimir_collate_fn([item] * 10, tokenizer)
    masked, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)

    # Sequence should be masked
    assert (labels_seq != -100).any()

    # Structure labels should all be -100 (no supervision)
    assert (labels_struct == -100).all()


# --- compute_mlm_loss ---


def test_compute_mlm_loss(tokenizer):
    """Test: compute_mlm_loss correctly calculates weighted loss, NaN exclusions, and metrics."""
    B, L, V_seq, V_struct = 2, 4, 64, 4096

    seq_logits = torch.zeros(B, L, V_seq)
    struct_logits = torch.zeros(B, L, V_struct)

    labels_seq = torch.full((B, L), -100, dtype=torch.long)
    labels_struct = torch.full((B, L), -100, dtype=torch.long)

    # Batch 0: 2 seq masked + 1 struct masked + 1 struct NaN (excluded)
    labels_seq[0, 1] = 10; seq_logits[0, 1, 10] = 100.0
    labels_seq[0, 2] = 20; seq_logits[0, 2, 5] = 100.0

    labels_struct[0, 2] = 500; struct_logits[0, 2, 500] = 100.0
    labels_struct[0, 3] = tokenizer.struct_nan  # NaN — must be excluded

    # Batch 1: 1 seq masked only
    labels_seq[1, 1] = 30; seq_logits[1, 1, 30] = 100.0

    lam = 0.5
    loss, sample_loss, metrics = compute_mlm_loss(
        sequence_logits=seq_logits,
        structure_logits=struct_logits,
        labels_seq=labels_seq,
        labels_struct=labels_struct,
        tokenizer=tokenizer,
        lam=lam,
        gradient_accumulation_steps=2,
    )

    # Metric counts
    assert metrics["full_seq_total"] == 2
    assert metrics["full_seq_correct"] == 1
    assert metrics["full_struct_total"] == 1
    assert metrics["full_struct_correct"] == 1
    assert metrics["partial_seq_total"] == 1
    assert metrics["partial_seq_correct"] == 1
    assert metrics["overall_total"] == 4
    assert metrics["overall_correct"] == 3

    # NaN exclusion
    mask_struct_valid = (labels_struct != -100) & (labels_struct != tokenizer.struct_nan)
    assert mask_struct_valid[0, 3].item() is False

    # Boosted loss math
    expected_w0 = 1.0 + 0.5 * math.log(4)  # 1 + lam * log(1 + 3)
    expected_w1 = 1.0 + 0.5 * math.log(2)  # 1 + lam * log(1 + 1)

    boosted_0 = expected_w0 * sample_loss[0]
    boosted_1 = expected_w1 * sample_loss[1]
    expected_final = (boosted_0 + boosted_1) / 2.0 / 2.0  # mean / grad_accum

    assert torch.allclose(loss, expected_final, atol=1e-4)


# --- _compute_detailed_metrics ---


def test_detailed_metrics_full_supervision(tokenizer):
    """Test: _compute_detailed_metrics correctly routes a sample with structure supervision to 'full' buckets."""
    B, L = 1, 4
    V_seq, V_struct = 64, 4200

    labels_seq = torch.full((B, L), -100, dtype=torch.long)
    labels_struct = torch.full((B, L), -100, dtype=torch.long)

    # 2 masked seq positions, 1 masked struct position
    labels_seq[0, 0] = 5
    labels_seq[0, 1] = 10
    labels_struct[0, 2] = 500

    seq_logits = torch.zeros(B, L, V_seq)
    struct_logits = torch.zeros(B, L, V_struct)
    seq_logits[0, 0, 5] = 100.0  # correct
    seq_logits[0, 1, 3] = 100.0  # wrong
    struct_logits[0, 2, 500] = 100.0  # correct

    mask_seq = labels_seq != -100
    mask_struct_valid = (labels_struct != -100) & (labels_struct != tokenizer.struct_nan)

    num_masked_seq = mask_seq.sum(dim=1).float()
    num_masked_struct = mask_struct_valid.sum(dim=1).float()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss_seq = criterion(seq_logits.view(-1, V_seq), labels_seq.view(-1)).view(B, L)
    loss_struct = criterion(struct_logits.view(-1, V_struct), labels_struct.view(-1)).view(B, L)
    sample_loss_seq = (loss_seq * mask_seq.float()).sum(dim=1)
    sample_loss_struct = (loss_struct * mask_struct_valid.float()).sum(dim=1)

    m = _compute_detailed_metrics(
        labels_seq, labels_struct, seq_logits, struct_logits,
        mask_seq, mask_struct_valid, num_masked_seq, num_masked_struct,
        sample_loss_seq, sample_loss_struct,
    )

    # Has struct supervision → routed to "full" buckets
    assert m["full_seq_total"] == 2
    assert m["full_seq_correct"] == 1
    assert m["full_struct_total"] == 1
    assert m["full_struct_correct"] == 1
    assert m["partial_seq_total"] == 0
    assert m["overall_total"] == 3
    assert m["overall_correct"] == 2


def test_detailed_metrics_partial_supervision(tokenizer):
    """Test: _compute_detailed_metrics routes a sample without structure supervision to 'partial' buckets."""
    B, L = 1, 4
    V_seq, V_struct = 64, 4200

    labels_seq = torch.full((B, L), -100, dtype=torch.long)
    labels_struct = torch.full((B, L), -100, dtype=torch.long)

    # 1 masked seq position, no struct
    labels_seq[0, 1] = 10

    seq_logits = torch.zeros(B, L, V_seq)
    struct_logits = torch.zeros(B, L, V_struct)
    seq_logits[0, 1, 10] = 100.0  # correct

    mask_seq = labels_seq != -100
    mask_struct_valid = (labels_struct != -100) & (labels_struct != tokenizer.struct_nan)

    num_masked_seq = mask_seq.sum(dim=1).float()
    num_masked_struct = mask_struct_valid.sum(dim=1).float()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss_seq = criterion(seq_logits.view(-1, V_seq), labels_seq.view(-1)).view(B, L)
    loss_struct = criterion(struct_logits.view(-1, V_struct), labels_struct.view(-1)).view(B, L)
    sample_loss_seq = (loss_seq * mask_seq.float()).sum(dim=1)
    sample_loss_struct = (loss_struct * mask_struct_valid.float()).sum(dim=1)

    m = _compute_detailed_metrics(
        labels_seq, labels_struct, seq_logits, struct_logits,
        mask_seq, mask_struct_valid, num_masked_seq, num_masked_struct,
        sample_loss_seq, sample_loss_struct,
    )

    # No struct supervision → routed to "partial" buckets
    assert m["partial_seq_total"] == 1
    assert m["partial_seq_correct"] == 1
    assert m["full_seq_total"] == 0
    assert m["full_struct_total"] == 0
    assert m["overall_total"] == 1
    assert m["overall_correct"] == 1

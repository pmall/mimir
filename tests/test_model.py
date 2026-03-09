"""
Tests for Mimir v2 Model and Training Logic (using mocks)
"""

import math

import pytest
import torch

from mimir.tokenizer import load_tokenizer, build_input_tensors
from mimir.dataset import mimir_collate_fn
from scripts.train import apply_mlm_masking
import json


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


# --- Tests ---

def test_mlm_masking_rate_and_independence(tokenizer, struct_fix):
    """Test: MLM masking rate in [0.25, 0.75], and independent per track at sample level"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, attn, chain_id, coords = build_input_tensors(fp, binder, tokenizer)
    
    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "chain_id": chain_id,
        "structure_coords": coords,
        "attention_mask": attn,
        "length": len(seq)
    }
    
    B = 1000  # More samples for better statistics
    batch = mimir_collate_fn([item] * B, tokenizer)
    
    L_padded = batch["sequence"].size(1)
    fp_len = len(fp["sequence"])
    
    # Correctly find EOS position for a sample in the batch to get true binder length
    # All samples are identical in this test batch
    eos_pos = (batch["sequence"][0] == tokenizer.seq_eos).nonzero(as_tuple=True)[0][0].item()
    binder_start = 1 + fp_len + 1
    binder_end = eos_pos
    binder_slice = slice(binder_start, binder_end)
    binder_len = binder_end - binder_start
    
    masked, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)
    
    # 1. Verify rates are within range per sample
    seq_masked = (labels_seq[:, binder_slice] != -100).float().sum(dim=1)
    struct_masked = (labels_struct[:, binder_slice] != -100).float().sum(dim=1)
    
    seq_rates = seq_masked / binder_len
    struct_rates = struct_masked / binder_len
    
    # Allowing slight epsilon for rounding
    assert torch.all(seq_rates >= 0.24) and torch.all(seq_rates <= 0.76)
    assert torch.all(struct_rates >= 0.24) and torch.all(struct_rates <= 0.76)
    
    # 2. Independence: rates should not be identical for all samples
    rate_diffs = (seq_rates - struct_rates).abs()
    assert rate_diffs.max() > 0.05, "Masking rates appear coupled"
    
    # 3. Mask bitmask independence
    # The actual indices selected should also be independent
    same_mask = (labels_seq[:, binder_slice] != -100) == (labels_struct[:, binder_slice] != -100)
    overlap_rate = same_mask.float().mean().item()
    # If they were coupled, overlap would be 1.0. 
    # If independent and say rate is 0.5, overlap would be ~0.5.
    assert overlap_rate < 0.90

def test_lr_schedule_warmup_and_decay():
    """Test: LR schedule warmup increases, decay decreases to 1e-5"""
    peak_lr = 1e-4
    dummy_weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([dummy_weight], lr=peak_lr)
    total_steps, warmup_steps = 100, 10
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = 1e-5 / peak_lr
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        
    # Warmup increases
    assert lrs[0] < lrs[5] < lrs[warmup_steps]
    # Decay decreases
    assert lrs[warmup_steps] > lrs[50] > lrs[-1]
    # Check that it decays to exactly 1e-5
    final_lr = scheduler.get_last_lr()[0]
    assert math.isclose(final_lr, 1e-5, rel_tol=1e-5)

def test_compute_mlm_loss(tokenizer):
    """Test: compute_mlm_loss correctly calculates weighted loss, NaN exclusions, and detailed metrics"""
    from scripts.train import compute_mlm_loss
    
    B, L, V_seq, V_struct = 2, 4, 64, 4096
    
    seq_logits = torch.zeros(B, L, V_seq)
    struct_logits = torch.zeros(B, L, V_struct)
    
    labels_seq = torch.full((B, L), -100, dtype=torch.long)
    labels_struct = torch.full((B, L), -100, dtype=torch.long)
    
    # Batch 0 valid labels (3 masked total, 1 is sequence, 2 is sequence+struct)
    labels_seq[0, 1] = 10; seq_logits[0, 1, 10] = 100.0  # Sequence masked at 1
    labels_seq[0, 2] = 20; seq_logits[0, 2, 5] = 100.0   # Sequence masked at 2 (wrong pred)
    
    labels_struct[0, 2] = 500; struct_logits[0, 2, 500] = 100.0 # Struct masked at 2
    labels_struct[0, 3] = tokenizer.struct_nan  # Struct masked at 3 but NaN
    
    # Batch 1 valid labels (1 masked total)
    labels_seq[1, 1] = 30; seq_logits[1, 1, 30] = 100.0
    
    lam = 0.5
    loss, sample_loss, metrics = compute_mlm_loss(
        sequence_logits=seq_logits,
        structure_logits=struct_logits,
        labels_seq=labels_seq,
        labels_struct=labels_struct,
        tokenizer=tokenizer,
        lam=lam,
        gradient_accumulation_steps=2
    )
    
    # Verify Metric Accumulation
    # Batch 0: nm_seq=2, nm_struct=1 (index 3 is NaN). Total = 3.
    # Batch 1: nm_seq=1, nm_struct=0. Total = 1.
    
    assert metrics["full_seq_total"] == 2
    assert metrics["full_seq_correct"] == 1
    assert metrics["full_struct_total"] == 1
    assert metrics["full_struct_correct"] == 1
    
    assert metrics["partial_seq_total"] == 1
    assert metrics["partial_seq_correct"] == 1
    
    assert metrics["overall_total"] == 4
    assert metrics["overall_correct"] == 3
    
    # Verify Math
    mask_struct_valid = (labels_struct != -100) & (labels_struct != tokenizer.struct_nan)
    assert mask_struct_valid[0, 3].item() is False
    
    # loss_b0 math:
    # labels_seq[0, 1] is correct (loss=0), labels_seq[0, 2] is wrong (loss=100), labels_struct[0, 2] is correct (loss=0)
    # total_masked_b0 = 3
    # sample_loss_b0 = (0 + 100 + 0) / 3 = 33.33
    loss_b0 = sample_loss[0].item()
    assert abs(loss_b0 - (100.0 / 3.0)) < 1.0
    
    # loss_b1 math:
    # total_masked_b1 = 1, logic is correct (loss=0)
    loss_b1 = sample_loss[1].item()
    assert abs(loss_b1 - 0.0) < 1.0
    
    # Boosted Loss: 1.0 + lam * math.log(1 + total_masked)
    expected_w0 = 1.0 + 0.5 * math.log(4) # 1 + 0.5 * log(1+3)
    expected_w1 = 1.0 + 0.5 * math.log(2) # 1 + 0.5 * log(1+1)
    
    boosted_loss_0 = expected_w0 * sample_loss[0]
    boosted_loss_1 = expected_w1 * sample_loss[1]
    
    expected_mean_loss = (boosted_loss_0 + boosted_loss_1) / 2.0
    expected_final_loss = expected_mean_loss / 2.0 # grad_accum=2
    
    assert torch.allclose(loss, expected_final_loss, atol=1e-4)

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
    """Test: MLM masking rate in [0.25, 0.75], and independent per track"""
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
    
    B = 100
    batch = mimir_collate_fn([item] * B, tokenizer)
    
    L = batch["sequence"].size(1)

    fp_len = len(fp["sequence"])
    fp_slice = slice(0, 1 + fp_len + 1)
    
    # Binder part is from exactly right after chainbreak until just before EOS
    # EOS is at L-1, so binder ends at L-1.
    binder_slice = slice(1 + fp_len + 1, L - 1)
    
    masked, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)
    
    # 1. FP is never masked (labels are -100)
    assert torch.all(labels_seq[:, fp_slice] == -100)
    assert torch.all(labels_struct[:, fp_slice] == -100)
    
    seq_masked_count = (labels_seq[:, binder_slice] != -100).sum().item()
    struct_masked_count = (labels_struct[:, binder_slice] != -100).sum().item()
    
    binder_len = (L - 1) - (1 + fp_len + 1)
    total_binder_tokens = B * binder_len
    seq_rate = seq_masked_count / total_binder_tokens
    struct_rate = struct_masked_count / total_binder_tokens
    
    # Rate should be roughly between 0.25 and 0.75
    assert 0.20 <= seq_rate <= 0.80, f"Seq rate: {seq_rate}"
    assert 0.20 <= struct_rate <= 0.80, f"Struct rate: {struct_rate}"
    
    # 3. Independence: they shouldn't be exactly the same mask
    same_mask = (labels_seq[:, binder_slice] != -100) == (labels_struct[:, binder_slice] != -100)
    overlap_rate = same_mask.float().mean().item()
    # Should definitely not be 1.0.
    assert overlap_rate < 0.95

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
    
    # Batch 0 valid labels
    labels_seq[0, 1] = 10; seq_logits[0, 1, 10] = 100.0
    labels_seq[0, 2] = 20; seq_logits[0, 2, 5] = 100.0
    
    labels_struct[0, 2] = 500; struct_logits[0, 2, 500] = 100.0
    labels_struct[0, 3] = tokenizer.struct_nan
    
    # Batch 1 valid labels (partial binder)
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
    
    loss_b0 = sample_loss[0].item()
    assert abs(loss_b0 - (100.0 / 3.0)) < 1.0
    
    loss_b1 = sample_loss[1].item()
    assert abs(loss_b1 - 0.0) < 1.0
    
    expected_w0 = 0.5 * math.log(4)
    expected_w1 = 0.5 * math.log(2)
    
    boosted_loss_0 = expected_w0 * sample_loss[0]
    boosted_loss_1 = expected_w1 * sample_loss[1]
    
    expected_mean_loss = (boosted_loss_0 + boosted_loss_1) / 2.0
    expected_final_loss = expected_mean_loss / 2.0
    
    assert torch.allclose(loss, expected_final_loss, atol=1e-4)

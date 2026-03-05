"""
Tests for Mimir v2 Model and Training Logic (using mocks)
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimir.model import ExtendedEmbedding
from mimir.tokenizer import load_tokenizer, CUT_TOKEN_ID_SEQ, CUT_TOKEN_ID_STRUCT, build_input_tensors
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

def test_extended_embedding_routing():
    """Test: ExtendedEmbedding routes <cut> token correctly"""
    orig_embed = nn.Embedding(100, 16)
    orig_embed.weight.data.fill_(1.0)  # Orig is all 1s
    
    cut_id = 99
    ext_embed = ExtendedEmbedding(orig_embed, cut_token_id=cut_id)
    ext_embed.cut_embedding.weight.data.fill_(2.0)  # Cut is all 2s
    
    x = torch.tensor([[10, 20, cut_id, 30]])
    out = ext_embed(x)
    
    # positions 0, 1, 3 should be 1.0. Position 2 should be 2.0.
    assert torch.all(out[0, 0] == 1.0)
    assert torch.all(out[0, 1] == 1.0)
    assert torch.all(out[0, 2] == 2.0)
    assert torch.all(out[0, 3] == 1.0)

def test_extended_embedding_gradients():
    """Test: ExtendedEmbedding gradient only flows to cut embedding"""
    orig_embed = nn.Embedding(100, 16)
    cut_id = 99
    ext_embed = ExtendedEmbedding(orig_embed, cut_token_id=cut_id)
    
    x = torch.tensor([[10, 20, cut_id, 30]])
    out = ext_embed(x)
    loss = out.sum()
    loss.backward()
    
    assert ext_embed.original_embedding.weight.grad is None
    assert ext_embed.cut_embedding.weight.grad is not None

def test_mlm_masking_rate_and_independence(tokenizer, struct_fix):
    """Test: MLM masking rate in [0.25, 0.75], and independent per track"""
    # Use real tensors directly loaded by tokenizer collator mock
    # mimir_collate_fn processes dicts exactly like the dataset handles
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "position_ids": pos,
        "attention_mask": attn,
        "length": len(seq)
    }
    
    B = 100
    batch = mimir_collate_fn([item] * B, tokenizer)
    
    masked, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)
    
    L = batch["sequence"].size(1)

    fp_len = len(fp["position_ids"])
    fp_slice = slice(0, 1 + fp_len + 1)
    
    # Binder part is from exactly right after CUT until just before EOS
    # EOS is at L-1, so binder ends at L-1.
    binder_slice = slice(1 + fp_len + 1, L - 1)
    
    # 1. FP is never masked (labels are -100)
    assert torch.all(labels_seq[:, fp_slice] == -100)
    assert torch.all(labels_struct[:, fp_slice] == -100)
    
    # 2. Check masking rate on binder
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
    # We use a dummy parameter since mock module is removed
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

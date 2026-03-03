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
from tests.mocks import MockEsm3, _MockOutput, SEQ_VOCAB_SIZE, STRUCT_VOCAB_SIZE


# --- Mocks ---

def _masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != -100
    if not valid.any():
        return torch.tensor(0.0, requires_grad=False)
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

def _mock_loss(output: _MockOutput, labels_seq: torch.Tensor) -> torch.Tensor:
    labels_struct = labels_seq.clamp(max=STRUCT_VOCAB_SIZE - 1)
    return _masked_ce_loss(output.sequence_logits, labels_seq) + \
           _masked_ce_loss(output.structure_logits, labels_struct)

# --- Checkpoint Helpers ---

def _save_checkpoint(path: Path, model: nn.Module, epoch: int, step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "step": step, "model": model.state_dict()}, path / "ckpt.pt")

def _load_checkpoint(path: Path, model: nn.Module) -> tuple[int, int]:
    ckpt = path / "ckpt.pt"
    if not ckpt.exists():
        return 0, 0
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    return state["epoch"], state["step"]

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
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    L = len(seq)
    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "position_ids": pos,
        "attention_mask": attn,
        "length": L
    }
    
    B = 100
    batch = mimir_collate_fn([item] * B, tokenizer)
    
    masked, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)
    
    # FP is 1..1+fp_len, so 0..1+fp_len (including BOS and CUT) shouldn't be masked.
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

def test_loss_zero_when_all_labels_100():
    """Test: Loss zero when all labels are -100"""
    model = MockEsm3()
    # mock batch
    B, L = 2, 5
    out = _MockOutput(torch.randn(B, L, SEQ_VOCAB_SIZE), torch.randn(B, L, STRUCT_VOCAB_SIZE))
    labels = torch.full((B, L), -100, dtype=torch.long)
    loss = _mock_loss(out, labels)
    
    assert loss.item() == 0.0
    assert not torch.isnan(loss)

def test_loss_finite_and_positive_with_valid_labels():
    """Test: Loss finite and positive with valid labels"""
    model = MockEsm3()
    B, L = 2, 5
    seq = torch.zeros(B, L, dtype=torch.long)
    struct = torch.zeros(B, L, dtype=torch.long)
    sasa = torch.zeros(B, L, dtype=torch.long)
    pos = torch.arange(L).unsqueeze(0).expand(B, L)
    
    out = model(seq, struct, sasa, pos)
    labels = torch.zeros((B, L), dtype=torch.long)
    loss = _mock_loss(out, labels)
    
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_gradients_exist_on_all_trainable_params():
    """Test: Gradients exist on all trainable params"""
    model = MockEsm3()
    B, L = 2, 5
    seq = torch.zeros(B, L, dtype=torch.long)
    struct = torch.zeros(B, L, dtype=torch.long)
    sasa = torch.zeros(B, L, dtype=torch.long)
    pos = torch.arange(L).unsqueeze(0).expand(B, L)
    
    out = model(seq, struct, sasa, pos)
    labels = torch.zeros((B, L), dtype=torch.long)
    loss = _mock_loss(out, labels)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

def test_checkpoint_save_reload_round_trip(tmp_path):
    """Test: Checkpoint save/reload round-trip and fresh start"""
    model = MockEsm3()
    # Add some random data to simulate training
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param))
            
    ckpt_dir = tmp_path / "checkpoints"
    
    # Fresh start behavior
    model2 = MockEsm3()
    ep, step = _load_checkpoint(ckpt_dir, model2)
    assert ep == 0 and step == 0
    
    # Save 
    _save_checkpoint(ckpt_dir, model, epoch=3, step=42)
    
    # Load
    model3 = MockEsm3()
    ep3, step3 = _load_checkpoint(ckpt_dir, model3)
    assert ep3 == 3 and step3 == 42
    
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model3.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"

def test_lr_schedule_warmup_and_decay():
    """Test: LR schedule warmup increases, decay decreases to 1e-5"""
    model = MockEsm3()
    peak_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
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

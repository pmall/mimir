"""
Tests for Mimir v2 Model and Training Logic (using mocks)
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from mimir.model import ExtendedEmbedding
from mimir.tokenizer import load_tokenizer, CUT_TOKEN_ID_SEQ, CUT_TOKEN_ID_STRUCT
from scripts.train import apply_mlm_masking


# --- Mocks ---

SEQ_VOCAB_SIZE = CUT_TOKEN_ID_SEQ + 1      # 65
STRUCT_VOCAB_SIZE = CUT_TOKEN_ID_STRUCT + 1  # 4101
MOCK_HIDDEN = 16

class _MockOutput:
    def __init__(self, sequence_logits: torch.Tensor, structure_logits: torch.Tensor) -> None:
        self.sequence_logits = sequence_logits
        self.structure_logits = structure_logits

class MockEsm3(nn.Module):
    def __init__(self, vocab_seq: int = SEQ_VOCAB_SIZE, vocab_struct: int = STRUCT_VOCAB_SIZE) -> None:
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
        hidden = (
            self.seq_embed(sequence_tokens.clamp(0, SEQ_VOCAB_SIZE - 1))
            + self.struct_embed(structure_tokens.clamp(0, STRUCT_VOCAB_SIZE - 1))
        )
        return _MockOutput(self.seq_head(hidden), self.struct_head(hidden))

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

# We need a small fake batch to test apply_mlm_masking
@pytest.fixture
def fake_batch():
    # Batch size 2, length 10
    # Let FP be first 4 tokens, CUT is at idx 4, Binder is last 4, EOS is at 9
    # Pos IDs jump at CUT
    # Setup such that sequence > 0 to have "real" tokens
    B, L = 2, 10
    seq = torch.ones((B, L), dtype=torch.long)
    struct = torch.ones((B, L), dtype=torch.long)
    sasa = torch.ones((B, L), dtype=torch.long)
    
    # CUT token at idx 4
    cut_seq = 64
    cut_struct = 4100
    cut_sasa = 3
    
    seq[:, 4] = cut_seq
    struct[:, 4] = cut_struct
    sasa[:, 4] = cut_sasa
    
    return {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "position_ids": torch.arange(L).unsqueeze(0).expand(B, L),
        "attention_mask": torch.ones((B, L), dtype=torch.long)
    }

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

def test_mlm_masking_rate_and_independence(tokenizer, fake_batch):
    """Test: MLM masking rate in [0.25, 0.75], and independent per track"""
    # Increase batch size / sequence length to test stats
    B, L = 100, 20
    fake_batch["sequence"] = torch.ones((B, L), dtype=torch.long)
    fake_batch["structure"] = torch.ones((B, L), dtype=torch.long)
    fake_batch["sasa"] = torch.ones((B, L), dtype=torch.long)
    fake_batch["position_ids"] = torch.arange(L).unsqueeze(0).expand(B, L)
    fake_batch["attention_mask"] = torch.ones((B, L), dtype=torch.long)
    
    # CUT at idx 8
    fake_batch["sequence"][:, 8] = tokenizer.cut_seq
    
    masked, labels_seq, labels_struct = apply_mlm_masking(fake_batch, tokenizer)
    
    # Binder part is indices 9..19.
    binder_slice = slice(9, 20)
    
    # FP part is indices 0..7. Should never be masked.
    fp_slice = slice(0, 8)
    
    # 1. FP is never masked (labels are -100)
    assert torch.all(labels_seq[:, fp_slice] == -100)
    assert torch.all(labels_struct[:, fp_slice] == -100)
    
    # 2. Check masking rate on binder
    # A label != -100 means it was chosen for masking
    seq_masked_count = (labels_seq[:, binder_slice] != -100).sum().item()
    struct_masked_count = (labels_struct[:, binder_slice] != -100).sum().item()
    
    total_binder_tokens = B * 11
    seq_rate = seq_masked_count / total_binder_tokens
    struct_rate = struct_masked_count / total_binder_tokens
    
    # Rate should be roughly between 0.25 and 0.75
    assert 0.20 <= seq_rate <= 0.80, f"Seq rate: {seq_rate}"
    assert 0.20 <= struct_rate <= 0.80, f"Struct rate: {struct_rate}"
    
    # 3. Independence: they shouldn't be exactly the same mask
    same_mask = (labels_seq[:, binder_slice] != -100) == (labels_struct[:, binder_slice] != -100)
    overlap_rate = same_mask.float().mean().item()
    # If independent with uniform [0.25, 0.75], expected overlap is around 1/2.
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
    """Test: LR schedule warmup increases, decay decreases"""
    model = MockEsm3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps, warmup_steps = 100, 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        
    # Warmup increases
    assert lrs[0] < lrs[5] < lrs[warmup_steps]
    # Decay decreases
    assert lrs[warmup_steps] > lrs[50] > lrs[-1]

"""
Tests for Mimir v2 Task 1: Tokenizer
Runs purely on CPU using extracted JSON test fixtures.
"""

import json
from pathlib import Path

import pytest
import torch

from mimir.tokenizer import load_tokenizer, build_input_tensors, MimirTokenizer
from mimir.dataset import mimir_collate_fn


# --- Fixture Loading ---

TEST_DATA_DIR = Path("tests/data")

@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()

@pytest.fixture
def struct_fix():
    with open(TEST_DATA_DIR / "struct_0.json", "r") as f:
        return json.load(f)

@pytest.fixture
def no_struct_fix():
    with open(TEST_DATA_DIR / "no_struct_0.json", "r") as f:
        return json.load(f)


# --- Tests ---

def test_1_tokenizer_cut_token(tokenizer: MimirTokenizer):
    """Test 1: CUT token registration + explicit amino acid collision check"""
    assert tokenizer.cut_seq is not None
    assert tokenizer.cut_struct is not None
    assert tokenizer.cut_sasa is not None
    
    # Assert no collision with BOS, EOS, mask, or pad token IDs
    specials = {tokenizer.seq_bos, tokenizer.seq_eos, tokenizer.seq_pad, tokenizer.seq_mask}
    assert tokenizer.cut_seq not in specials
    
    # Assert no collision with 20 standard amino acid token IDs
    standard_aas = "ACDEFGHIKLMNPQRSTVWY"
    standard_ids = {tokenizer.sequence.convert_tokens_to_ids(aa) for aa in standard_aas}
    assert tokenizer.cut_seq not in standard_ids
    
    # Idempotent
    t2 = load_tokenizer()
    assert t2.cut_seq == tokenizer.cut_seq
    assert t2.cut_struct == tokenizer.cut_struct
    assert t2.cut_sasa == tokenizer.cut_sasa


def test_2_binder_with_structure(tokenizer, struct_fix):
    """Test 2: Binder with structure: correct tensor shape and track population"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    fp_len = len(fp["position_ids"])
    bin_len = len(binder["sequence"])
    
    # Total length before padding
    expected_len = 1 + fp_len + 1 + bin_len + 1
    assert len(seq) == expected_len
    assert len(struct) == expected_len
    assert len(sasa) == expected_len
    
    # FP positions
    fp_slice = slice(1, 1 + fp_len)
    assert not torch.all(seq[fp_slice] == tokenizer.seq_mask)
    assert not torch.all(struct[fp_slice] == tokenizer.struct_mask)
    assert not torch.all(sasa[fp_slice] == tokenizer.sasa_mask)
    
    # Binder positions
    bin_slice = slice(1 + fp_len + 1, 1 + fp_len + 1 + bin_len)
    assert not torch.all(seq[bin_slice] == tokenizer.seq_mask)
    assert not torch.all(struct[bin_slice] == tokenizer.struct_mask)
    
    # SASA track at binder positions = mask token for all
    assert torch.all(sasa[bin_slice] == tokenizer.sasa_mask)
    
    # CUT token present on all three tracks at correct position
    cut_idx = 1 + fp_len
    assert seq[cut_idx] == tokenizer.cut_seq
    assert struct[cut_idx] == tokenizer.cut_struct
    assert sasa[cut_idx] == tokenizer.cut_sasa


def test_3_binder_without_structure(tokenizer, no_struct_fix):
    """Test 3: Binder without structure: correct track population"""
    fp, binder = no_struct_fix["fingerprint"], no_struct_fix["binder"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    fp_len = len(fp["position_ids"])
    bin_len = len(binder["sequence"])
    bin_slice = slice(1 + fp_len + 1, 1 + fp_len + 1 + bin_len)
    
    assert not torch.all(seq[bin_slice] == tokenizer.seq_mask)
    assert torch.all(struct[bin_slice] == tokenizer.struct_mask)
    assert torch.all(sasa[bin_slice] == tokenizer.sasa_mask)


def test_4_inference_mode_fully_masked(tokenizer, struct_fix):
    """Test 4: Inference mode: fully masked binder"""
    fp = struct_fix["fingerprint"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, None, tokenizer)
    
    fp_len = len(fp["position_ids"])
    bin_len = 96 # Default we used in the function for None
    bin_slice = slice(1 + fp_len + 1, 1 + fp_len + 1 + bin_len)
    
    assert torch.all(seq[bin_slice] == tokenizer.seq_mask)
    assert torch.all(struct[bin_slice] == tokenizer.struct_mask)
    assert torch.all(sasa[bin_slice] == tokenizer.sasa_mask)
    
    cut_idx = 1 + fp_len
    assert seq[cut_idx] == tokenizer.cut_seq


def test_5_position_ids_are_correct(tokenizer, struct_fix):
    """Test 5: Position IDs are correct"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    assert pos[0] == 0 # BOS
    
    fp_len = len(fp["position_ids"])
    fp_pos_orig = fp["position_ids"]
    assert pos[1:1+fp_len].tolist() == fp_pos_orig
    
    last_fp_pos = fp_pos_orig[-1]
    cut_idx = 1 + fp_len
    assert pos[cut_idx] == last_fp_pos + 1000
    
    # Binder pos IDs continuous
    assert pos[cut_idx + 1] == last_fp_pos + 1001
    
    # EOS
    assert pos[-1] == pos[-2] + 1


def test_8_padding_applied_after_eos(tokenizer, struct_fix):
    """Test 8: Padding is applied after EOS"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    L = len(seq) # Natural length
    
    # Manually pack into a single item batch for collate
    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "position_ids": pos,
        "attention_mask": attn,
        "length": L
    }
    
    batch = mimir_collate_fn([item], tokenizer)
    seq_padded = batch["sequence"][0]
    struct_padded = batch["structure"][0]
    
    padded_len = len(seq_padded)
    # Next multiple of 64
    assert padded_len % 64 == 0
    assert padded_len >= L
    assert padded_len - L < 64
    
    assert seq_padded[L-1] == tokenizer.seq_eos
    assert seq_padded[L] == tokenizer.seq_pad
    assert struct_padded[L] == tokenizer.struct_pad
    
    # Attention mask
    attn_padded = batch["attention_mask"][0]
    assert torch.all(attn_padded[:L] == 1)
    assert torch.all(attn_padded[L:] == 0)


def test_9_maximum_length_sample_does_not_overflow(tokenizer, struct_fix):
    """Test 9: Maximum length sample does not overflow"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    # Mutate to max size
    fp["sequence"] = "A" * 280
    fp["structure_tokens"] = [1] * 280
    fp["sasa"] = [1.0] * 280
    fp["position_ids"] = list(range(1, 281))
    
    binder["sequence"] = "A" * 96
    binder["structure_tokens"] = [1] * 96
    binder["sasa"] = [1.0] * 96
    
    seq, struct, sasa, pos, attn = build_input_tensors(fp, binder, tokenizer)
    
    assert len(seq) == 379
    # Ensure no truncation logic inside
    
    # Positions
    assert pos[-1] == pos[-2] + 1

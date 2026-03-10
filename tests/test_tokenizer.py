"""
Tests for Mimir v2 Tokenizer and Dataloader.
Runs purely on CPU using extracted JSON test fixtures.
"""

import json
from pathlib import Path

import pytest
import torch

from mimir.tokenizer import load_tokenizer, build_input_tensors, MimirTokenizer
from mimir.dataset import MimirDataset, mimir_collate_fn


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

def test_tokenizer_chainbreak_token(tokenizer: MimirTokenizer):
    """Test 1: Chainbreak token registration + explicit amino acid collision check"""
    # Spec: SEQUENCE_CHAINBREAK = 31
    assert tokenizer.seq_chainbreak == 31
    # Spec: STRUCTURE_CHAINBREAK = 4100
    assert tokenizer.struct_chainbreak == 4100
    # Spec: SASA_PAD = 0 (used for chainbreak too)
    assert tokenizer.sasa_chainbreak == 0
    
    # Spec: STRUCTURE_MASK = 4096
    assert tokenizer.struct_mask == 4096
    # Spec: Undefined structure (nan coords) = 2246
    assert tokenizer.struct_nan == 2246
    
    # Assert no collision with BOS, EOS, mask, or pad token IDs
    specials = {tokenizer.seq_bos, tokenizer.seq_eos, tokenizer.seq_pad, tokenizer.seq_mask}
    assert tokenizer.seq_chainbreak not in specials
    
    # Assert no collision with 20 standard amino acid token IDs
    standard_aas = "ACDEFGHIKLMNPQRSTVWY"
    standard_ids = {tokenizer.sequence.convert_tokens_to_ids(aa) for aa in standard_aas}
    assert tokenizer.seq_chainbreak not in standard_ids
    
    # Idempotent
    t2 = load_tokenizer()
    assert t2.seq_chainbreak == tokenizer.seq_chainbreak
    assert t2.struct_chainbreak == tokenizer.struct_chainbreak
    assert t2.sasa_chainbreak == tokenizer.sasa_chainbreak


def test_binder_with_structure(tokenizer, struct_fix):
    """Test 2: Binder with structure: correct tensor shape and track population"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, attn, chain_id, coords = build_input_tensors(fp, binder, tokenizer)
    
    fp_len = len(fp["sequence"])
    bin_len = len(binder["sequence"])
    
    # Total length before padding
    expected_len = 1 + fp_len + 1 + bin_len + 1
    assert len(seq) == expected_len
    assert len(struct) == expected_len
    assert len(sasa) == expected_len
    assert len(chain_id) == expected_len
    assert coords.shape == (expected_len, 3, 3)
    
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
    
    # Chainbreak token present on all three tracks at correct position
    cut_idx = 1 + fp_len
    assert seq[cut_idx] == tokenizer.seq_chainbreak
    assert struct[cut_idx] == tokenizer.struct_chainbreak
    assert sasa[cut_idx] == tokenizer.sasa_chainbreak
    
    # Chain ID: 1 for fingerprint + chainbreak, 2 for binder + EOS
    assert torch.all(chain_id[:cut_idx + 1] == 1)  # BOS + FP + chainbreak
    assert torch.all(chain_id[cut_idx + 1:] == 2)  # binder + EOS
    
    # Structure coords: real values for fingerprint, NaN elsewhere
    assert not torch.isnan(coords[1:1 + fp_len]).any()
    assert torch.isnan(coords[0]).all()  # BOS
    assert torch.isnan(coords[cut_idx]).all()  # chainbreak
    assert torch.isnan(coords[cut_idx + 1:-1]).all()  # binder
    assert torch.isnan(coords[-1]).all()  # EOS


def test_binder_without_structure(tokenizer, no_struct_fix):
    """Test 3: Binder without structure: correct track population"""
    fp, binder = no_struct_fix["fingerprint"], no_struct_fix["binder"]
    seq, struct, sasa, attn, chain_id, coords = build_input_tensors(fp, binder, tokenizer)
    
    fp_len = len(fp["sequence"])
    bin_len = len(binder["sequence"])
    bin_slice = slice(1 + fp_len + 1, 1 + fp_len + 1 + bin_len)
    
    assert not torch.all(seq[bin_slice] == tokenizer.seq_mask)
    assert torch.all(struct[bin_slice] == tokenizer.struct_mask)
    assert torch.all(sasa[bin_slice] == tokenizer.sasa_mask)


def test_inference_mode_fully_masked(tokenizer, struct_fix):
    """Test 4: Inference mode: fully masked binder"""
    fp = struct_fix["fingerprint"]
    seq, struct, sasa, sequence_id, chain_id, coords = build_input_tensors(fp, None, tokenizer)
    
    fp_len = len(fp["sequence"])
    bin_len = 96  # Default we used in the function for None
    bin_slice = slice(1 + fp_len + 1, 1 + fp_len + 1 + bin_len)
    
    assert torch.all(seq[bin_slice] == tokenizer.seq_mask)
    assert torch.all(struct[bin_slice] == tokenizer.struct_mask)
    assert torch.all(sasa[bin_slice] == tokenizer.sasa_mask)
    
    cut_idx = 1 + fp_len
    assert seq[cut_idx] == tokenizer.seq_chainbreak


def test_padding_applied_after_eos(tokenizer, struct_fix):
    """Test 5: Padding is applied after EOS"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, sequence_id, chain_id, coords = build_input_tensors(fp, binder, tokenizer)
    
    L = len(seq)
    
    # Manually pack into a single item batch for collate
    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "sequence_id": sequence_id,
        "chain_id": chain_id,
        "structure_coords": coords,
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
    
    assert seq_padded[L - 1] == tokenizer.seq_eos
    assert seq_padded[L] == tokenizer.seq_pad
    assert struct_padded[L] == tokenizer.struct_pad
    
    # Sequence ID
    sequence_id_padded = batch["sequence_id"][0]
    assert torch.all(sequence_id_padded[:L] == 1)
    assert torch.all(sequence_id_padded[L:] == 0)
    
    # Chain ID padded with 0
    chain_id_padded = batch["chain_id"][0]
    assert chain_id_padded[L] == 0


def test_maximum_length_sample_does_not_overflow(tokenizer, struct_fix):
    """Test 6: Maximum length sample does not overflow"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    # Mutate to max size
    fp["sequence"] = "A" * 280
    fp["structure_tokens"] = [1] * 280
    fp["sasa"] = [0.5] * 280
    fp["coordinates"] = [[[0, 0, 0], [1.5, 0, 0], [2, 1, 0]]] * 280
    
    binder["sequence"] = "A" * 96
    binder["structure_tokens"] = [1] * 96
    binder["sasa"] = [0.5] * 96
    
    seq, struct, sasa, sequence_id, chain_id, coords = build_input_tensors(fp, binder, tokenizer)
    
    assert len(seq) == 379


def test_dataset_loads_correctly(tokenizer):
    """Test 7: MimirDataset loads data correctly with new fields"""
    dataset = MimirDataset(
        associations_csv=TEST_DATA_DIR / "associations.csv",
        fingerprints_lmdb=TEST_DATA_DIR / "fingerprints.lmdb",
        binders_lmdb=TEST_DATA_DIR / "binders.lmdb",
        tokenizer=tokenizer,
    )
    
    assert len(dataset) > 0
    
    # Get first valid sample
    sample = dataset[0]
    assert sample is not None
    assert "sequence" in sample
    assert "structure" in sample
    assert "sasa" in sample
    assert "sequence_id" in sample
    assert "chain_id" in sample
    assert "structure_coords" in sample
    assert "length" in sample
    
    # Verify shapes
    L = sample["length"]
    assert sample["chain_id"].shape[0] == L
    assert sample["structure_coords"].shape == (L, 3, 3)


def test_collate_handles_chain_id_and_coords(tokenizer, struct_fix):
    """Test 8: Collate function handles chain_id and structure_coords correctly"""
    fp, binder = struct_fix["fingerprint"], struct_fix["binder"]
    seq, struct, sasa, sequence_id, chain_id, coords = build_input_tensors(fp, binder, tokenizer)
    
    L = len(seq)
    item = {
        "sequence": seq,
        "structure": struct,
        "sasa": sasa,
        "sequence_id": sequence_id,
        "chain_id": chain_id,
        "structure_coords": coords,
        "length": L
    }
    
    batch = mimir_collate_fn([item], tokenizer)
    
    # Check chain_id padding - padded to next multiple of 64
    padded_len = ((L + 63) // 64) * 64
    assert batch["chain_id"].shape == (1, padded_len)
    assert batch["chain_id"][0, L] == 0  # padded value
    
    # Check coords padding
    assert batch["structure_coords"].shape == (1, padded_len, 3, 3)
    assert torch.isnan(batch["structure_coords"][0, L]).all()

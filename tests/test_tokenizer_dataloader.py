"""
Tests for Mimir v2 Task 1: Tokenizer & Dataloader
Runs purely on CPU using extracted JSON test fixtures.
"""

import json
from pathlib import Path
import lmdb
import msgpack

import pytest
import torch
from torch.utils.data import DataLoader

from mimir.tokenizer import load_tokenizer, build_input_tensors, MimirTokenizer
from mimir.dataset import MimirDataset, BucketBatchSampler, mimir_collate_fn


# --- Fixture Loading ---

TEST_DATA_DIR = Path("tests/data")

@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()

@pytest.fixture
def struct_fix(request):
    with open(TEST_DATA_DIR / f"struct_0.json", "r") as f:
        return json.load(f)

@pytest.fixture
def no_struct_fix(request):
    with open(TEST_DATA_DIR / f"no_struct_0.json", "r") as f:
        return json.load(f)

@pytest.fixture
def missing_fp_fix():
    with open(TEST_DATA_DIR / "missing_fp_0.json", "r") as f:
        return json.load(f)

@pytest.fixture
def missing_binder_fix():
    with open(TEST_DATA_DIR / "missing_binder_0.json", "r") as f:
        return json.load(f)


# --- Tests ---

def test_1_tokenizer_cut_token(tokenizer: MimirTokenizer):
    """Test 1: CUT token registration"""
    assert tokenizer.cut_seq is not None
    assert tokenizer.cut_struct is not None
    assert tokenizer.cut_sasa is not None
    
    # Assert no collision with BOS, EOS, mask, or pad token IDs
    specials = {tokenizer.seq_bos, tokenizer.seq_eos, tokenizer.seq_pad, tokenizer.seq_mask}
    assert tokenizer.cut_seq not in specials
    
    # Assert no collision with 20 standard amino acid token IDs (0 to 32 roughly, CUT is 64)
    # The lowest we use is CUT_SEQ=64, so it's > 32
    assert tokenizer.cut_seq >= 33
    
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


# Mock a minimal Dataset to test skip logic
# The Dataset takes paths, we will make a fake CSV and fake LMDB with only 1 item to test missing keys

@pytest.fixture
def fake_env(tmp_path):
    # Create fake associations
    assocs = tmp_path / "assocs.csv"
    with open(assocs, "w") as f:
        f.write("target,binder_id\nFP1,B1\nFP2,B2\nFP3,B3\n")
        
    fp_db = tmp_path / "fps"
    fp_env = lmdb.open(str(fp_db), map_size=1048576)
    with fp_env.begin(write=True) as txn:
        # Provide FP1 and FP2, missing FP3
        txn.put(b"FP1", msgpack.packb({"sequence":"A", "structure_tokens":[1], "sasa":[1.0], "position_ids":[1]}))
        txn.put(b"FP2", msgpack.packb({"sequence":"A", "structure_tokens":[1], "sasa":[1.0], "position_ids":[1]}))
    
    bin_db = tmp_path / "bins"
    bin_env = lmdb.open(str(bin_db), map_size=1048576)
    with bin_env.begin(write=True) as txn:
        # Provide B1 and B3, missing B2
        txn.put(b"B1", msgpack.packb({"sequence":"A", "structure_tokens":[1], "sasa":[1.0]}))
        txn.put(b"B3", msgpack.packb({"sequence":"A", "structure_tokens":[1], "sasa":[1.0]}))
        
    return assocs, fp_db, bin_db

def test_6_and_7_missing_keys_skip_silently(tokenizer, fake_env):
    """Test 6 & 7: Missing fingerprint or binder key: sample is skipped"""
    assocs, fp_db, bin_db = fake_env
    ds = MimirDataset(assocs, fp_db, bin_db, tokenizer)
    
    assert len(ds) == 3
    # First is FP1/B1 - both present -> not None
    assert ds[0] is not None
    # Second is FP2/B2 - B2 missing -> None (Test 7)
    assert ds[1] is None
    # Third is FP3/B3 - FP3 missing -> None (Test 6)
    assert ds[2] is None
    
    # Batching them through collate function
    def wrap_collate(b):
        return mimir_collate_fn(b, tokenizer)
        
    dl = DataLoader(ds, batch_size=3, collate_fn=wrap_collate)
    batch = next(iter(dl))
    # Batch should only contain the 1 valid sample padded to multiple of 64
    assert batch["sequence"].shape[0] == 1


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


def test_10_bucket_batching_minimizes_padding(tokenizer, fake_env):
    """Test 10: Bucket batching minimizes padding"""
    assocs, fp_db, bin_db = fake_env
    # First item is L=4 (1+1+1+1) -> bucket 64
    # Let's add more items by mocking lengths dynamically in Sampler
    ds = MimirDataset(assocs, fp_db, bin_db, tokenizer)
    
    # Mock more lengths
    sampler = BucketBatchSampler(ds, batch_size=2)
    # Mock natural lengths: two short (64s), two long (128s)
    sampler.lengths = [10, 20, 80, 90]
    
    buckets = sampler._get_buckets()
    # Should place 10,20 into bucket 64. 80,90 into bucket 128
    assert len(buckets[64]) == 2
    assert len(buckets[128]) == 2
    
    batches = list(iter(sampler))
    # Expect 2 batches of size 2, grouped together
    assert any(set(b) == {0, 1} for b in batches)
    assert any(set(b) == {2, 3} for b in batches)


def test_11_epoch_shuffling_is_reproducible(fake_env, tokenizer):
    """Test 11: Epoch shuffling is reproducible"""
    assocs, fp_db, bin_db = fake_env
    ds = MimirDataset(assocs, fp_db, bin_db, tokenizer)
    
    sampler1 = BucketBatchSampler(ds, batch_size=2, epoch=42)
    sampler1.lengths = [1, 2, 3, 4, 5, 6]
    
    sampler2 = BucketBatchSampler(ds, batch_size=2, epoch=42)
    sampler2.lengths = [1, 2, 3, 4, 5, 6]
    
    sampler3 = BucketBatchSampler(ds, batch_size=2, epoch=99)
    sampler3.lengths = [1, 2, 3, 4, 5, 6]
    
    batches1 = list(iter(sampler1))
    batches2 = list(iter(sampler2))
    batches3 = list(iter(sampler3))
    
    assert batches1 == batches2
    assert batches1 != batches3

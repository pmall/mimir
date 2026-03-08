"""
Tests for Mimir v2 Task 1: Dataloader
Consumes only pre-generated fixtures from tests/data/.
Run `scripts/extract_test_data.py` to regenerate the fixtures.
"""

from pathlib import Path

import pytest

from mimir.tokenizer import load_tokenizer
from mimir.dataset import MimirDataset, BucketBatchSampler, mimir_collate_fn


# --- Constants ---

TEST_DATA_DIR = Path("tests/data")
ASSOCS_CSV = TEST_DATA_DIR / "associations.csv"
FP_LMDB = TEST_DATA_DIR / "fingerprints.lmdb"
BIN_LMDB = TEST_DATA_DIR / "binders.lmdb"


# --- Fixtures ---

@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="session")
def dataset(tokenizer):
    return MimirDataset(ASSOCS_CSV, FP_LMDB, BIN_LMDB, tokenizer)


# --- Tests ---

def test_missing_fingerprint_skip(dataset):
    """Test: Missing fingerprint key: sample is skipped.

    The associations.csv has a row whose fingerprint key is absent from
    fingerprints.lmdb (index 4, the missing_fp row). MimirDataset must
    return None for it.
    """
    # Row 4 (0-indexed) is the missing-fp case
    assert dataset[4] is None


def test_missing_binder_skip(dataset):
    """Test: Missing binder key: sample is skipped.

    The associations.csv has a row whose binder_id is absent from
    binders.lmdb (index 5, the missing_binder row). MimirDataset must
    return None for it.
    """
    # Row 5 (0-indexed) is the missing-binder case
    assert dataset[5] is None


def test_collate_filters_nones(tokenizer, dataset):
    """Test: mimir_collate_fn filters None entries, only valid items survive."""
    # Fetch the 4 valid + 2 invalid samples and collate them together
    items = [dataset[i] for i in range(len(dataset))]
    batch = mimir_collate_fn(items, tokenizer)
    # Only the 4 valid samples should be in the batch
    assert batch["sequence"].shape[0] == 4


def test_bucket_batching_groups_similar_lengths(tokenizer, dataset):
    """Test: BucketBatchSampler groups items into batches from the same bucket."""
    sampler = BucketBatchSampler(dataset, batch_size=2)
    buckets = sampler._get_buckets()

    batches = list(iter(sampler))
    assert len(batches) > 0

    for batch in batches:
        # Every index in the batch must belong to the same bucket
        batch_set = set(batch)
        containing_buckets = {
            bucket_id
            for bucket_id, members in buckets.items()
            if batch_set & set(members)
        }
        assert len(containing_buckets) == 1, (
            f"Batch {batch} spans multiple buckets: {containing_buckets}"
        )


def test_epoch_shuffling_is_reproducible(tokenizer, dataset):
    """Test: Same epoch seed → same batch order; different seed → different order."""
    sampler_a1 = BucketBatchSampler(dataset, batch_size=2, epoch=42)
    sampler_a2 = BucketBatchSampler(dataset, batch_size=2, epoch=42)
    sampler_b = BucketBatchSampler(dataset, batch_size=2, epoch=99)

    batches_a1 = list(iter(sampler_a1))
    batches_a2 = list(iter(sampler_a2))
    batches_b = list(iter(sampler_b))

    assert batches_a1 == batches_a2
    assert batches_a1 != batches_b

def test_bucket_batching_less_padding_than_random(tokenizer, dataset):
    """Test: Bucket batching produces fewer or equal total pad tokens than random batching."""
    import random
    
    # 1. Bucket batching padding
    bucket_sampler = BucketBatchSampler(dataset, batch_size=2)
    bucket_batches = list(iter(bucket_sampler))
    
    bucket_pad_tokens = 0
    for batch_indices in bucket_batches:
        items = [dataset[i] for i in batch_indices]
        batch = mimir_collate_fn(items, tokenizer)
        L = batch["attention_mask"].sum(dim=1)
        pad = batch["sequence"].shape[1] * len(L) - L.sum().item()
        bucket_pad_tokens += pad
        
    # 2. Random batching padding
    valid_indices = [i for i, size in enumerate(bucket_sampler.lengths) if size > 0]
    random.seed(42)
    shuffled_indices = valid_indices.copy()
    random.shuffle(shuffled_indices)
    
    random_batches = [shuffled_indices[i:i+2] for i in range(0, len(shuffled_indices), 2)]
    
    random_pad_tokens = 0
    for batch_indices in random_batches:
        items = [dataset[i] for i in batch_indices]
        batch = mimir_collate_fn(items, tokenizer)
        L = batch["attention_mask"].sum(dim=1)
        pad = batch["sequence"].shape[1] * len(L) - L.sum().item()
        random_pad_tokens += pad
        
    # Assert
    assert bucket_pad_tokens <= random_pad_tokens

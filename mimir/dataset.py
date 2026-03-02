"""
PyTorch Dataset and Dataloader utilities for Mimir v2.
Handles LMDB reading, bucket-based batching, and dynamic padding.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple, Optional
import math
import random

import lmdb
import msgpack
import torch
from torch.utils.data import Dataset, Sampler

from mimir.tokenizer import MimirTokenizer, build_input_tensors

logger = logging.getLogger(__name__)


def pad_to_multiple(length: int, multiple: int = 64) -> int:
    """Returns the nearest multiple of `multiple` above or equal to `length`."""
    if length == 0:
        return multiple
    return math.ceil(length / multiple) * multiple


class MimirDataset(Dataset):
    """
    Dataset for Mimir v2 fine-tuning.
    Reads associations from CSV and streams features from LMDBs.
    """

    def __init__(
        self,
        associations_csv: Path | str,
        fingerprints_lmdb: Path | str,
        binders_lmdb: Path | str,
        tokenizer: MimirTokenizer,
    ):
        self.associations_csv = Path(associations_csv)
        self.fingerprints_lmdb = Path(fingerprints_lmdb)
        self.binders_lmdb = Path(binders_lmdb)
        self.tokenizer = tokenizer
        
        # We will lazy-initialize lmdb environments in worker processes
        self._fp_env = None
        self._bin_env = None
        
        self.samples: List[Dict[str, str]] = []
        self._load_associations()
        
    def _load_associations(self):
        """Loads all associations into memory."""
        with open(self.associations_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support varying column names
                target = row.get("target_id", row.get("target", row.get("uniprot_accession")))
                binder = row.get("binder_id")
                if target and binder:
                    self.samples.append({
                        "target": target,
                        "binder": binder
                    })
        logger.info(f"Loaded {len(self.samples)} associations from {self.associations_csv}")

    def _init_lmdbs(self):
        if self._fp_env is None:
            self._fp_env = lmdb.open(str(self.fingerprints_lmdb), readonly=True, lock=False)
        if self._bin_env is None:
            self._bin_env = lmdb.open(str(self.binders_lmdb), readonly=True, lock=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Returns tensor dict for the sample or None if missing from LMDBs.
        Dataloader collate_fn must cleanly filter out Nones.
        """
        self._init_lmdbs()
        
        sample = self.samples[idx]
        target_id = sample["target"]
        binder_id = sample["binder"]
        
        with self._fp_env.begin() as txn:
            fp_data = txn.get(target_id.encode("utf-8"))
        if not fp_data:
            return None
            
        with self._bin_env.begin() as txn:
            bin_data = txn.get(binder_id.encode("utf-8"))
        if not bin_data:
            return None
            
        fp_obj = msgpack.unpackb(fp_data, raw=False)
        bin_obj = msgpack.unpackb(bin_data, raw=False)
        
        seq, struct, sasa, pos_ids, attn_mask = build_input_tensors(
            fingerprint=fp_obj,
            binder=bin_obj,
            tokenizer=self.tokenizer
        )
        
        return {
            "sequence": seq,
            "structure": struct,
            "sasa": sasa,
            "position_ids": pos_ids,
            "attention_mask": attn_mask,
            "length": len(seq)
        }


def mimir_collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]], tokenizer: MimirTokenizer) -> Dict[str, torch.Tensor]:
    """
    Filters None (skipped samples) and pads the batch to a multiple of 64.
    """
    valid_batch = [b for b in batch if b is not None]
    
    if not valid_batch:
        # Edge case: entire batch is skipped
        return {}
        
    max_len = max(b["length"] for b in valid_batch)
    padded_len = pad_to_multiple(max_len, 64)
    
    batch_size = len(valid_batch)
    
    # Initialize padded tensors
    seq_padded = torch.full((batch_size, padded_len), tokenizer.seq_pad, dtype=torch.long)
    struct_padded = torch.full((batch_size, padded_len), tokenizer.struct_pad, dtype=torch.long)
    sasa_padded = torch.full((batch_size, padded_len), tokenizer.sasa_pad, dtype=torch.long)
    # Position IDs can be padded with 0 since they are ignored by attention mask
    pos_padded = torch.zeros((batch_size, padded_len), dtype=torch.long)
    # Attention mask padded with 0
    attn_padded = torch.zeros((batch_size, padded_len), dtype=torch.long)
    
    for i, item in enumerate(valid_batch):
        l = item["length"]
        seq_padded[i, :l] = item["sequence"]
        struct_padded[i, :l] = item["structure"]
        sasa_padded[i, :l] = item["sasa"]
        pos_padded[i, :l] = item["position_ids"]
        attn_padded[i, :l] = item["attention_mask"]
        
    return {
        "sequence": seq_padded,
        "structure": struct_padded,
        "sasa": sasa_padded,
        "position_ids": pos_padded,
        "attention_mask": attn_padded,
    }


class BucketBatchSampler(Sampler):
    """
    Groups samples of similar lengths into batches to minimize padding.
    """
    def __init__(
        self,
        dataset: MimirDataset,
        batch_size: int,
        epoch: int = 0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        
        # We need lengths for bucketing. Since lengths depend on FP length (in LMDB),
        # getting exact length for every sample upfront is slow (minutes for large DBs).
        # We scan it once efficiently or rely on a pre-computed lengths dictionary.
        # However, for correct bucket batching, we must scan at init.
        self.lengths = self._scan_lengths()
        
    def _scan_lengths(self) -> List[int]:
        logger.info("Scanning dataset lengths for bucket batching...")
        lengths = []
        fp_env = lmdb.open(str(self.dataset.fingerprints_lmdb), readonly=True, lock=False)
        bin_env = lmdb.open(str(self.dataset.binders_lmdb), readonly=True, lock=False)
        
        skipped = 0
        with fp_env.begin() as fp_txn, bin_env.begin() as bin_txn:
            for sample in self.dataset.samples:
                fp_data = fp_txn.get(sample['target'].encode('utf-8'))
                bin_data = bin_txn.get(sample['binder'].encode('utf-8'))
                
                if not fp_data or not bin_data:
                    lengths.append(-1) # Placeholder for missing
                    skipped += 1
                    continue
                    
                fp_obj = msgpack.unpackb(fp_data, raw=False)
                bin_obj = msgpack.unpackb(bin_data, raw=False)
                
                # Total length = 1 (BOS) + FP length + 1 (CUT) + Binder length + 1 (EOS)
                fp_len = len(fp_obj["position_ids"])
                bin_len = len(bin_obj["sequence"])
                total_len = 1 + fp_len + 1 + bin_len + 1
                lengths.append(total_len)
                
        if skipped > 0:
            logger.info(f"Skipped {skipped} samples during length scan (missing from LMDB).")
            
        fp_env.close()
        bin_env.close()
        return lengths

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        # Define buckets based on multiples of 64
        buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(self.lengths):
            if length == -1:
                # We can still batch missing samples; they'll be filtered by collate_fn
                # We assign them to the smallest bucket to minimize impact
                bucket_id = 64
            else:
                bucket_id = pad_to_multiple(length, 64)
                
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(idx)
            
        # Shuffle within each bucket with the epoch seed
        rng = random.Random(self.epoch)
        for b_id in buckets:
            rng.shuffle(buckets[b_id])
            
        # Form batches
        batches = []
        for b_id, indices in buckets.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
                elif len(batch) > 0:
                    # Drop last or keep? Usually we keep partial batches unless drop_last=True
                    batches.append(batch)
                    
        # Shuffle the order of batches
        rng.shuffle(batches)
        
        return iter(batches)

    def __len__(self) -> int:
        # Approximate if there are partial batches
        total_batches = 0
        for b_id, indices in self._get_buckets().items():
            total_batches += math.ceil(len(indices) / self.batch_size)
        return total_batches

    def _get_buckets(self):
        # Helper to calculate length without modifying state
        buckets: Dict[int, List[int]] = {}
        for idx, length in enumerate(self.lengths):
            bucket_id = 64 if length == -1 else pad_to_multiple(length, 64)
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(idx)
        return buckets

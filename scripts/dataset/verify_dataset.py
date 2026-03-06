"""
Verify dataloader execution across the entire dataset ensuring 0 formatting panics during batch generation.

Usage:
    uv run python scripts/dataset/verify_dataset.py --config data/run78-v2/config.json
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimir.config import load_config
from mimir.dataset import MimirDataset, BucketBatchSampler, mimir_collate_fn
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)

def verify_dataloader(name: str, csv_path: Path, fp_lmdb: Path, binders_lmdb: Path, tokenizer, batch_size=32) -> bool:
    logger.info(f"--- Verifying Dataloader: {name} ---")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Fingerprints: {fp_lmdb}")
    logger.info(f"Binders: {binders_lmdb}")

    if not csv_path.exists():
        logger.error(f"Missing CSV: {csv_path}")
        return False

    try:
        dataset = MimirDataset(
            associations_csv=csv_path,
            fingerprints_lmdb=fp_lmdb,
            binders_lmdb=binders_lmdb,
            tokenizer=tokenizer
        )
        
        logger.info(f"Total samples defined in CSV: {len(dataset)}")
        
        sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            epoch=0
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: mimir_collate_fn(b, tokenizer),
            num_workers=0, # Keep to 0 for strict local sequential tracebacks
            pin_memory=False
        )
    except Exception as e:
        logger.error(f"Failed to initialize DataLoader for {name}: {e}")
        return False

    total_batches = 0
    total_samples = 0
    total_skipped = 0

    try:
        for batch in tqdm(dataloader, desc=f"Scanning {name} batches"):
            total_batches += 1
            
            # The collate_fn filters out Nones resulting in fewer valid elements.
            seq = batch["sequence"]
            struct = batch["structure"]
            sasa = batch["sasa"]
            pos = batch["position_ids"]
            attn = batch["attention_mask"]
            skipped = int(batch["num_skipped"].item())
            
            valid_batch_size = seq.shape[0]
            total_samples += valid_batch_size
            total_skipped += skipped
            
            # If the entire batch was skipped, tensors are size (0, 0)
            if valid_batch_size == 0:
                continue
                
            # Verify explicit tensor dimensional constraints.
            assert seq.dim() == 2, f"Expected 2D Sequence Tensor, got {seq.dim()}"
            assert struct.dim() == 2, f"Expected 2D Structure Tensor, got {struct.dim()}"
            assert sasa.dim() == 2, f"Expected 2D SASA Tensor, got {sasa.dim()}"
            assert pos.dim() == 2, f"Expected 2D Pos Tensor, got {pos.dim()}"
            assert attn.dim() == 2, f"Expected 2D Attn Tensor, got {attn.dim()}"
            
            # Shapes must match exactly.
            b, l = seq.shape
            assert struct.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs Struct {struct.shape}"
            assert sasa.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs SASA {sasa.shape}"
            assert pos.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs Pos {pos.shape}"
            assert attn.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs Attn {attn.shape}"
            
            # Tokenizer dimension mappings must evaluate normally.
            assert struct.dtype == torch.long
            assert seq.dtype == torch.long
            
            # --- Advanced Format and Values Checks ---
            
            # 1. Monotonically increasing position IDs
            # Wait, position IDs are padded with 0. 
            # We must only check the sequence up to the valid length or use the attention mask.
            for i in range(b):
                valid_len = attn[i].sum().item()
                if valid_len <= 1:
                    continue
                    
                valid_positions = pos[i, :valid_len]
                diffs = valid_positions[1:] - valid_positions[:-1]
                assert torch.all(diffs > 0), f"Position IDs must be strictly monotonically increasing. Found backwards or repeating pos: {valid_positions}"
                
                # 2. Check +1000 gap at the cut.
                # Cut token is tokenizer.cut_seq (64). Find its index.
                cut_indices = (seq[i, :valid_len] == tokenizer.cut_seq).nonzero(as_tuple=True)[0]
                if len(cut_indices) > 0:
                    cut_idx = cut_indices[0].item()
                    if cut_idx > 0:
                        gap_at_cut = pos[i, cut_idx].item() - pos[i, cut_idx - 1].item()
                        assert gap_at_cut == 1000, f"Expected gap of EXACTLY 1000 at the cut token, but found gap {gap_at_cut} at index {cut_idx} for sequence {i}. Previous pos: {pos[i, cut_idx-1]}, Cut pos: {pos[i, cut_idx]}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"FATAL: Batch generation failed at batch {total_batches+1} -> {e}")
        return False

    logger.info(f"SUCCESS: {name} Dataloader verified.")
    logger.info(f"   Batches yielded:  {total_batches}")
    logger.info(f"   Valid samples:    {total_samples}")
    logger.info(f"   Skipped samples:  {total_skipped}")
    
    # Alert if too many items are skipped
    if total_samples == 0:
        logger.error(f"FAILURE: {name} Dataloader yielded ZERO valid samples! All {total_skipped} entries failed LMDB lookups.")
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify DataLoader End-to-End Extraction.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stdout,
    )
    
    config = load_config(args.config)
    
    logger.info("Loading tokenizer (this may take a few seconds)...")
    tokenizer = load_tokenizer()
    
    dataset_ok = verify_dataloader(
        name="Merged Dataset",
        csv_path=config.binders_merged,
        fp_lmdb=config.features_fingerprints,
        binders_lmdb=config.features_binders,
        tokenizer=tokenizer
    )
    
    if not dataset_ok:
        logger.error("Dataloader Pipeline Verification FAILED.")
        sys.exit(1)
        
    logger.info("Dataloader Pipeline Verification PASSED! Datasets are fully consistent for PyTorch training loops.")

if __name__ == "__main__":
    main()

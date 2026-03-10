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
            chain_id = batch["chain_id"]
            coords = batch["structure_coords"]
            attn = batch["sequence_id"]
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
            assert chain_id.dim() == 2, f"Expected 2D Chain ID Tensor, got {chain_id.dim()}"
            assert coords.dim() == 4, f"Expected 4D Coords Tensor, got {coords.dim()}"
            assert attn.dim() == 2, f"Expected 2D Attn Tensor, got {attn.dim()}"
            
            # Shapes must match exactly.
            b, l = seq.shape
            assert struct.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs Struct {struct.shape}"
            assert sasa.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs SASA {sasa.shape}"
            assert chain_id.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs Chain ID {chain_id.shape}"
            assert coords.shape == (b, l, 3, 3), f"Shape mismatch Seq {seq.shape} vs Coords {coords.shape}"
            assert attn.shape == (b, l), f"Shape mismatch Seq {seq.shape} vs Attn {attn.shape}"
            
            # Tokenizer dimension mappings must evaluate normally.
            assert struct.dtype == torch.long
            assert seq.dtype == torch.long
            
            # --- Advanced Format and Values Checks ---
            
            # 1. Chain ID: must be 1 for fingerprint+chainbreak, 2 for binder+EOS
            for i in range(b):
                valid_len = attn[i].sum().item()
                if valid_len <= 1:
                    continue
                
                valid_chain_ids = chain_id[i, :valid_len]
                
                # Find chainbreak position (transition from 1 to 2)
                chainbreak_indices = (valid_chain_ids[1:] == 2) & (valid_chain_ids[:-1] == 1)
                if chainbreak_indices.any():
                    # There should be exactly one chainbreak
                    assert chainbreak_indices.sum() == 1, f"Expected exactly one chainbreak, found {chainbreak_indices.sum()}"
                
                # 2. Chainbreak token present at chain transition
                # Chainbreak token is tokenizer.seq_chainbreak (31). Find its index.
                chainbreak_token_indices = (seq[i, :valid_len] == tokenizer.seq_chainbreak).nonzero(as_tuple=True)[0]
                if len(chainbreak_token_indices) > 0:
                    cb_idx = chainbreak_token_indices[0].item()
                    # Chain ID at chainbreak position should be 1 (fingerprint side)
                    assert chain_id[i, cb_idx] == 1, f"Chainbreak token should be at chain_id=1 position, got {chain_id[i, cb_idx]}"
                    # Next position should be chain 2 (binder)
                    if cb_idx + 1 < valid_len:
                        assert chain_id[i, cb_idx + 1] == 2, f"Position after chainbreak should be chain 2, got {chain_id[i, cb_idx + 1]}"
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

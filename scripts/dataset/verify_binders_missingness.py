import argparse
import logging
import sys
from pathlib import Path

import lmdb
import msgpack
from mimir.config import load_config
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate distribution of missing structure coordinates (NaNs).")
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stdout,
    )
    
    config = load_config(args.config)
    tokenizer = load_tokenizer()
    
    if not config.features_binders.exists():
        logger.error(f"Features LMDB not found at {config.features_binders}")
        sys.exit(1)
        
    env = lmdb.open(str(config.features_binders), readonly=True, lock=False)
    
    total_structures = 0
    total_tokens = 0
    total_nan_tokens = 0
    
    missing_pcts = []
    
    with env.begin() as txn:
        for key, value in txn.cursor():
            data = msgpack.unpackb(value)
            
            struct = data.get("structure_tokens")
            if struct is None:
                continue
                
            total_structures += 1
            length = len(struct)
            nan_count = struct.count(tokenizer.struct_nan)
            
            total_tokens += length
            total_nan_tokens += nan_count
            
            missing_pct = nan_count / length if length > 0 else 0
            missing_pcts.append(missing_pct)
            
    if total_structures == 0:
        print("No structures found.")
        return
        
    global_missing_pct = (total_nan_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    # Categorize
    perfect = sum(1 for p in missing_pcts if p == 0)
    under_10 = sum(1 for p in missing_pcts if 0 < p <= 0.10)
    under_30 = sum(1 for p in missing_pcts if 0.10 < p <= 0.30)
    under_50 = sum(1 for p in missing_pcts if 0.30 < p <= 0.50)
    over_50 = sum(1 for p in missing_pcts if p > 0.50)
    
    print("\n--- Structure Missingness Distribution ---")
    print(f"Total Structures Analyzed: {total_structures}")
    print(f"Global Missingness: {global_missing_pct:.2f}% ({total_nan_tokens} / {total_tokens} tokens are NaN)")
    print("\nMissingness Per Sequence (NaN Token %):")
    print(f"  0% missing (Perfect):       {perfect:4d} structures ({(perfect/total_structures)*100:.1f}%)")
    print(f"  1% - 10% missing:           {under_10:4d} structures ({(under_10/total_structures)*100:.1f}%)")
    print(f" 10% - 30% missing:           {under_30:4d} structures ({(under_30/total_structures)*100:.1f}%)")
    print(f" 30% - 50% missing:           {under_50:4d} structures ({(under_50/total_structures)*100:.1f}%)")
    print(f" > 50% missing (Very poor):   {over_50:4d} structures ({(over_50/total_structures)*100:.1f}%)")
    print("------------------------------------------\n")


if __name__ == "__main__":
    main()

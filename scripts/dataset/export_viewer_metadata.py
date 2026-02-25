"""
Export human target fingerprints data into a CSV for the Next.js viewer.

Extracts pLDDT, rSASA, smoothed rSASA, mask, and fingerprint details
from the target features LMDB and saves them in a flat CSV.

Usage:
    uv run python -m scripts.dataset.export_viewer_metadata \
        -i data/run78-v2/features_targets.lmdb \
        -o data/viewer_data.csv \
        [--max-len 280] [--verbose]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from mimir.structure_features import (
    TargetFeatures,
    compute_rsasa,
    get_fingerprint_mask,
    get_smoothed_rsasa,
)

# ---
# Constants
# ---

logger = logging.getLogger(__name__)


# ---
# Export Logic
# ---


def export_viewer_metadata(
    input_lmdb: Path,
    output_csv: Path,
    max_len: int = 280,
    limit: int | None = None,
) -> None:
    """Read targets LMDB, compute properties, and write CSV.

    Args:
        input_lmdb: Path to the input targets LMDB.
        output_csv: Path to the output CSV file.
        max_len: Max allowed fingerprint length.
        limit: Optional maximum number of targets to process.
    """
    logger.info(f"Opening input LMDB: {input_lmdb}")

    input_env = lmdb.open(
        str(input_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_read": 0,
        "valid_fingerprints": 0,
        "skipped_min_length": 0,
    }

    # Prepare CSV
    fieldnames = [
        "target_id",
        "sequence",
        "positions",
        "mask",
        "rsasa",
        "smoothed_rsasa",
        "plddt",
        "rsasa_threshold",
    ]

    with input_env.begin() as in_txn:
        total_entries = in_txn.stat()["entries"]
        if limit:
            total_entries = min(total_entries, limit)

        pbar = tqdm(
            total=total_entries,
            desc="Exporting Metadata",
            unit="target",
            dynamic_ncols=True,
        )

        cursor = in_txn.cursor()

        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for _key, value in cursor:
                if limit and stats["total_read"] >= limit:
                    break
                
                stats["total_read"] += 1
                
                # De-serialize TargetFeatures
                raw_dict = msgpack.unpackb(value)
                target = TargetFeatures.from_dict(raw_dict)
                
                # Compute mask & threshold
                mask, threshold = get_fingerprint_mask(
                    sequence=target.sequence,
                    sasa=target.sasa,
                    plddt=target.residue_plddt,
                    max_len=max_len,
                )
                
                if mask is None:
                    stats["skipped_min_length"] += 1
                else:
                    stats["valid_fingerprints"] += 1
                    
                    # Compute underlying values for the viewer
                    rsasa = compute_rsasa(target.sequence, target.sasa)
                    smoothed_rsasa = get_smoothed_rsasa(rsasa)
                    
                    # Prepare row
                    row = {
                        "target_id": target.entry_id,
                        "sequence": target.sequence,
                        "positions": json.dumps(target.position_ids),
                        "mask": json.dumps(mask.tolist()),
                        "rsasa": json.dumps([round(x, 3) for x in rsasa.tolist()]),
                        "smoothed_rsasa": json.dumps([round(x, 3) for x in smoothed_rsasa.tolist()]),
                        "plddt": json.dumps([round(x, 2) for x in target.residue_plddt]),
                        "rsasa_threshold": threshold if threshold is not None else "",
                    }
                    
                    writer.writerow(row)
                
                pbar.update(1)

        pbar.close()

    input_env.close()

    # ---
    # Summary
    # ---
    logger.info("Processing complete.")
    logger.info(f"  Total targets read:      {stats['total_read']}")
    logger.info(f"  Valid fingerprints:      {stats['valid_fingerprints']}")
    logger.info(f"  Skipped (length < 15):   {stats['skipped_min_length']}")
    logger.info(f"  Exported to CSV:         {output_csv}")


# ---
# Main
# ---

def main() -> None:
    """Parse CLI arguments and initiate metadata export."""
    parser = argparse.ArgumentParser(
        description="Export Target LMDB to viewer CSV."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input features targets LMDB",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to the output csv file",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=280,
        help="Maximum length of the fingerprint (default: 280)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of entries to process (for testing)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Log progress and statistics",
    )
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.input.exists():
        logger.error(f"Input LMDB not found: {args.input}")
        sys.exit(1)

    export_viewer_metadata(
        input_lmdb=args.input,
        output_csv=args.output,
        max_len=args.max_len,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

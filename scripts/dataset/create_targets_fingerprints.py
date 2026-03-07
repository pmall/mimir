"""
Extract human target fingerprints from the extracted AlphaFold2 structures database.

Filters down positions from the target features based on pLDDT and relative SASA.
Maintains continuous synchronous tracks.

Usage:
    uv run python -m scripts.dataset.create_targets_fingerprints \\
        --config data/run78-v2/config.json \\
        [--max-len 280] [--limit N] [-v]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from mimir.config import load_config
from mimir.features import FingerprintFeatures, TargetFeatures, get_fingerprint_mask

# ---
# Constants
# ---

LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB virtual (sparse on Linux, no pre-alloc)


logger = logging.getLogger(__name__)


# ---
# Feature extraction
# ---


def extract_fingerprint(
    target: TargetFeatures,
    max_len: int = 280,
) -> FingerprintFeatures | None:
    """Extract a synchronized sub-sequence fingerprint matching masking rules.

    Args:
        target: The parsed Input Target Features.
        max_len: The maximum allowed length of the resulting filtered tracks.

    Returns:
        FingerprintFeatures if it passes the min length filter, otherwise None.
    """
    seq_np = np.array(list(target.sequence))
    tokens_np = np.array(target.structure_tokens)
    sasa_np = np.array(target.sasa)
    plddt_np = np.array(target.residue_plddt)
    coords_np = np.array(target.coordinates)
    
    # Generate 1-indexed position IDs
    pos_np = np.arange(1, len(target.sequence) + 1, dtype=int)

    # 1. Get the shared centralized boolean mask and threshold
    mask_result = get_fingerprint_mask(
        sequence=target.sequence,
        sasa=target.sasa,
        plddt=target.residue_plddt,
        max_len=max_len,
    )

    mask, threshold = mask_result

    if mask is None:
        return None
    
    # 2. Apply mask to all 6 synchronized tracks
    f_seq = seq_np[mask]
    f_tokens = tokens_np[mask]
    f_sasa = sasa_np[mask]
    f_plddt = plddt_np[mask]
    f_pos = pos_np[mask]
    f_coords = coords_np[mask]

    fingerprint = FingerprintFeatures(
        entry_id=target.entry_id,
        sequence="".join(f_seq.tolist()),
        structure_tokens=f_tokens.tolist(),
        sasa=f_sasa.tolist(),
        residue_plddt=f_plddt.tolist(),
        position_ids=f_pos.tolist(),
        coordinates=f_coords.tolist(),
        rsasa_threshold=threshold,
    )

    return fingerprint


# ---
# Main
# ---

def main() -> None:
    """Parse CLI arguments and initiate fingerprint extraction."""
    parser = argparse.ArgumentParser(
        description="Filter AlphaFold target features into fingerprints"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json",
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

    config = load_config(args.config)

    if not config.features_targets.exists():
        logger.error(f"Input LMDB not found: {config.features_targets}")
        sys.exit(1)

    create_targets_fingerprints(
        input_lmdb=config.features_targets,
        output_lmdb=config.features_fingerprints,
        max_len=args.max_len,
        limit=args.limit,
    )


def create_targets_fingerprints(
    input_lmdb: Path,
    output_lmdb: Path,
    max_len: int = 280,
    limit: int | None = None,
) -> None:
    """Process targets into fingerprints.

    Args:
        input_lmdb: Path to the input targets LMDB.
        output_lmdb: Path to the output fingerprints LMDB.
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

    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    output_env = lmdb.open(str(output_lmdb), map_size=LMDB_MAP_SIZE)

    stats = {
        "total_read": 0,
        "valid_fingerprints": 0,
        "skipped_min_length": 0,
    }

    with input_env.begin() as in_txn:
        total_entries = in_txn.stat()["entries"]
        if limit:
            total_entries = min(total_entries, limit)

        pbar = tqdm(
            total=total_entries,
            desc="Generating Fingerprints",
            unit="target",
            dynamic_ncols=True,
        )

        cursor = in_txn.cursor()
        for key, value in cursor:
            if limit and stats["total_read"] >= limit:
                break
            
            stats["total_read"] += 1
            
            # De-serialize TargetFeatures
            raw_dict = msgpack.unpackb(value)
            
            # Safely get the target features, TargetFeatures can gracefully construct
            target = TargetFeatures.from_dict(raw_dict)
            
            # Compute Fingerprint
            fingerprint = extract_fingerprint(target, max_len=max_len)
            
            if fingerprint is None:
                stats["skipped_min_length"] += 1
            else:
                stats["valid_fingerprints"] += 1
                with output_env.begin(write=True) as out_txn:
                    out_txn.put(key, msgpack.packb(fingerprint.to_dict()))
            
            pbar.update(1)

        pbar.close()

    input_env.close()
    output_env.close()

    # ---
    # Summary
    # ---
    logger.info("Processing complete.")
    logger.info(f"  Total targets read:      {stats['total_read']}")
    logger.info(f"  Valid fingerprints:      {stats['valid_fingerprints']}")
    logger.info(f"  Skipped (length < 15):   {stats['skipped_min_length']}")
    logger.info("")
    if stats["total_read"] > 0:
        logger.info(
            f"  Yield rate:              {stats['valid_fingerprints']}/{stats['total_read']} "
            f"({100 * stats['valid_fingerprints'] / stats['total_read']:.2f}%)"
        )


if __name__ == "__main__":
    main()

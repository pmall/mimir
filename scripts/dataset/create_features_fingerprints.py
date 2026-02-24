"""
Extract human target fingerprints from the extracted AlphaFold2 structures database.

Filters down positions from the target features based on pLDDT and relative SASA.
Maintains continuous synchronous tracks.

Usage:
    uv run python -m scripts.dataset.create_features_fingerprints \
        -i data/run78-v2/features_targets \
        -o data/run78-v2/features_fingerprints \
        [--max-len 157] [--verbose]
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

from mimir.structure_features import FingerprintFeatures, TargetFeatures

# ---
# Constants
# ---

LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB virtual (sparse on Linux, no pre-alloc)
MIN_FINGERPRINT_LEN = 15

# Tien et al. 2013 Maximum allowed SASA (Theoretical max in Gly-X-Gly). Used for rSASA.
MAX_SASA_REFERENCE = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
}


logger = logging.getLogger(__name__)


# ---
# Feature extraction
# ---

def compute_rsasa(sequence: str, sasa: list[float]) -> np.ndarray:
    """Compute Relative SASA arrays safely. Handle unknown residues by assuming minimum surface."""
    rsasa = np.zeros(len(sequence), dtype=np.float32)
    for i, (res, abs_sasa) in enumerate(zip(sequence, sasa)):
        max_sasa = MAX_SASA_REFERENCE.get(res.upper(), 1.0) # avoid division by 0
        rsasa[i] = abs_sasa / max_sasa
    return rsasa


def extract_fingerprint(
    target: TargetFeatures,
    max_len: int = 157,
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
    pos_np = np.array(target.position_ids)

    # 1. Compute rSASA
    rsasa_np = compute_rsasa(target.sequence, target.sasa)

    # 2. Boolean Mask: pLDDT >= 70 AND rSASA >= 0.15
    mask = (plddt_np >= 70.0) & (rsasa_np >= 0.15)
    
    # 3. Apply mask to all 5 synchronized tracks
    f_seq = seq_np[mask]
    f_tokens = tokens_np[mask]
    f_sasa = sasa_np[mask]
    f_plddt = plddt_np[mask]
    f_pos = pos_np[mask]
    f_rsasa = rsasa_np[mask]

    # 4. Global Length Filter Check
    current_length = len(f_seq)
    if current_length < MIN_FINGERPRINT_LEN:
        return None

    # 5. Length Truncation (Top rSASA Selection)
    if current_length > max_len:
        # Get indices of top `max_len` rsasa values
        # Note: np.argsort is ascending, so we take the last `max_len` elements
        top_indices = np.argsort(f_rsasa)[-max_len:]
        
        # Sort these indices to maintain chronological order
        chronological_indices = np.sort(top_indices)

        f_seq = f_seq[chronological_indices]
        f_tokens = f_tokens[chronological_indices]
        f_sasa = f_sasa[chronological_indices]
        f_plddt = f_plddt[chronological_indices]
        f_pos = f_pos[chronological_indices]

    fingerprint = FingerprintFeatures(
        entry_id=target.entry_id,
        sequence="".join(f_seq.tolist()),
        structure_tokens=f_tokens.tolist(),
        sasa=f_sasa.tolist(),
        residue_plddt=f_plddt.tolist(),
        position_ids=f_pos.tolist(),
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
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the input features targets LMDB",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to the output fingerprint LMDB",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=157,
        help="Maximum length of the fingerprint (default: 157)",
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

    create_features_fingerprints(
        input_lmdb=args.input,
        output_lmdb=args.output,
        max_len=args.max_len,
        limit=args.limit,
    )


def create_features_fingerprints(
    input_lmdb: Path,
    output_lmdb: Path,
    max_len: int = 157,
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

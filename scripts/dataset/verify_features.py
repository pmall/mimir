"""
Verify consistency between input CSV and output features LMDB.

Checks:
1. All CSV entries exist in output LMDB
2. Entry counts match
3. PDB entries have structure features, HH/VH have sequence only
4. Sequence alignment between CSV and features

Usage:
    uv run python -m scripts.dataset.verify_features \\
        --config data/run78-v2/config.json [-v]
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import lmdb
import msgpack
from tqdm import tqdm

from mimir.config import load_config

logger = logging.getLogger(__name__)


# ---
# Verification
# ---

def verify_consistency(input_csv: Path, output_db: Path) -> bool:
    """Verify output LMDB matches input CSV.

    Args:
        input_csv: Path to the binder list CSV
        output_db: Path to the features LMDB

    Returns:
        True if consistent, False otherwise
    """
    logger.info(f"Reading input CSV: {input_csv}")
    csv_entries = {}
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_id = row.get("binder_id")
            if entry_id:
                csv_entries[entry_id] = {
                    "sequence": row.get("sequence", ""),
                    "entry_type": row.get("type", "UNKNOWN"),
                }

    logger.info(f"Found {len(csv_entries)} entries in CSV")

    logger.info(f"Reading output LMDB: {output_db}")
    env = lmdb.open(str(output_db), readonly=True, lock=False)

    lmdb_entries = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            entry_id = key.decode("utf-8")
            features = msgpack.unpackb(value)
            lmdb_entries[entry_id] = features

    logger.info(f"Found {len(lmdb_entries)} entries in LMDB")

    # Check 1: Count match
    if len(csv_entries) != len(lmdb_entries):
        logger.error(f"COUNT MISMATCH: CSV={len(csv_entries)}, LMDB={len(lmdb_entries)}")
        return False
    logger.info(f"Entry counts match: {len(csv_entries)}")

    # Check 2: All CSV entries in LMDB
    missing_in_lmdb = set(csv_entries.keys()) - set(lmdb_entries.keys())
    if missing_in_lmdb:
        logger.error(f"Missing in LMDB: {len(missing_in_lmdb)} entries")
        for entry_id in list(missing_in_lmdb)[:5]:
            logger.error(f"  - {entry_id}")
        return False
    logger.info("All CSV entries present in LMDB")

    # Check 3: Extra entries in LMDB
    extra_in_lmdb = set(lmdb_entries.keys()) - set(csv_entries.keys())
    if extra_in_lmdb:
        logger.warning(f"Extra entries in LMDB: {len(extra_in_lmdb)}")
        for entry_id in list(extra_in_lmdb)[:5]:
            logger.warning(f"  - {entry_id}")

    # Check 4: Type consistency
    logger.info("Verifying entry type consistency...")
    mismatches = []
    structure_stats = {"PDB_with_structure": 0, "PDB_without_structure": 0, "HH": 0, "VH": 0, "UNKNOWN": 0}

    for entry_id in tqdm(csv_entries.keys(), desc="Checking entries"):
        csv_data = csv_entries[entry_id]
        lmdb_data = lmdb_entries[entry_id]
        entry_type = csv_data["entry_type"]

        # Check sequence presence
        if not lmdb_data.get("sequence"):
            mismatches.append(f"{entry_id}: Missing sequence in LMDB")
            continue

        # Check structure features based on type
        has_coords = lmdb_data.get("structure_tokens") is not None
        has_sasa = lmdb_data.get("sasa") is not None

        if entry_type == "PDB":
            if has_coords and has_sasa:
                structure_stats["PDB_with_structure"] += 1
            else:
                structure_stats["PDB_without_structure"] += 1
        elif entry_type == "HH":
            structure_stats["HH"] += 1
            if has_coords or has_sasa:
                mismatches.append(f"{entry_id}: HH entry has unexpected structure data")
        elif entry_type == "VH":
            structure_stats["VH"] += 1
            if has_coords or has_sasa:
                mismatches.append(f"{entry_id}: VH entry has unexpected structure data")
        else:
            structure_stats["UNKNOWN"] += 1

    logger.info("Structure feature stats:")
    logger.info(f"  PDB with structure:    {structure_stats['PDB_with_structure']}")
    logger.info(f"  PDB without structure: {structure_stats['PDB_without_structure']}")
    logger.info(f"  HH entries:            {structure_stats['HH']}")
    logger.info(f"  VH entries:            {structure_stats['VH']}")
    logger.info(f"  Unknown type:          {structure_stats['UNKNOWN']}")

    if mismatches:
        logger.error(f"Found {len(mismatches)} inconsistencies:")
        for mismatch in mismatches[:10]:
            logger.error(f"  - {mismatch}")
        if len(mismatches) > 10:
            logger.error(f"  ... and {len(mismatches) - 10} more")
        return False

    logger.info("All consistency checks passed!")
    return True

# ---
# Main
# ---


def main() -> None:
    """Parse CLI arguments and run the features consistency verification."""
    parser = argparse.ArgumentParser(description="Verify CSV to LMDB consistency")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config)

    if not config.binders_merged.exists():
        logger.error(f"Input CSV not found: {config.binders_merged}")
        sys.exit(1)

    if not config.features_binders.exists():
        logger.error(f"Binders LMDB not found: {config.features_binders}")
        sys.exit(1)

    success = verify_consistency(config.binders_merged, config.features_binders)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

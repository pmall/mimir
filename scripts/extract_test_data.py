"""
Script to extract real LMDB test data into JSON format for tokenizer/dataloader tests.

This script reads a provided associations CSV, looks up the target fingerprint and
binder features in their respective LMDBs, and extracts a small representative set
of examples (with structure, without structure, missing elements) into JSON files.

Usage:
    uv run python scripts/extract_test_data.py \
        --assocs-csv data/run78-v2/binders_lists/final_binders_96aa.csv \
        --fingerprints-db data/run78-v2/features_fingerprints \
        --binders-db data/run78-v2/features_binders \
        -o tests/data \
        [--verbose]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import lmdb
import msgpack

# ---
# Constants
# ---

logger = logging.getLogger(__name__)


# ---
# Feature extraction
# ---


def extract_test_data(
    assocs_csv: Path,
    fingerprints_db: Path,
    binders_db: Path,
    output_dir: Path,
) -> None:
    """Extracts test fixtures from LMDB databases into JSON files.
    
    Args:
        assocs_csv: Path to the CSV mapping targets to binders.
        fingerprints_db: Path to the target fingerprints LMDB.
        binders_db: Path to the binders features LMDB.
        output_dir: Output directory where JSON fixtures are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Opening LMDB environments...")
    fp_env = lmdb.open(str(fingerprints_db), readonly=True, lock=False)
    binders_env = lmdb.open(str(binders_db), readonly=True, lock=False)
    
    with fp_env.begin() as fp_txn, binders_env.begin() as bind_txn:
        struct_assocs = []
        no_struct_assocs = []
        missing_fp = []
        missing_binder = []
        
        logger.info(f"Scanning associations from {assocs_csv}...")
        with open(assocs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprot = row.get("target")
                binder_id = row.get("binder_id")
                
                # Check Fingerprint
                fp_data = fp_txn.get(uniprot.encode("utf-8")) if uniprot else None
                
                # Check Binder
                bind_data = bind_txn.get(binder_id.encode("utf-8")) if binder_id else None
                
                if fp_data and bind_data:
                    bind_obj = msgpack.unpackb(bind_data, raw=False)
                    fp_obj = msgpack.unpackb(fp_data, raw=False)
                    
                    if bind_obj.get("structure_tokens") is not None:
                        if len(struct_assocs) < 2:
                            struct_assocs.append({"assoc": row, "fingerprint": fp_obj, "binder": bind_obj})
                    else:
                        if len(no_struct_assocs) < 2:
                            no_struct_assocs.append({"assoc": row, "fingerprint": fp_obj, "binder": bind_obj})
                
                elif not fp_data and bind_data and uniprot:
                    if len(missing_fp) < 1:
                        missing_fp.append({"assoc": row})
                
                elif fp_data and not bind_data and binder_id:
                    if len(missing_binder) < 1:
                        missing_binder.append({"assoc": row})
                
                # Stop when we have everything needed for tests (2, 2, 1, 1 respectively)
                if len(struct_assocs) == 2 and len(no_struct_assocs) == 2 and len(missing_fp) == 1 and len(missing_binder) == 1:
                    break
                    
        # --- Write outputs ---
        def write_json(name: str, data: dict) -> None:
            path = output_dir / name
            with open(path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=2)
            logger.info(f"Saved: {path}")
                
        for i, data in enumerate(struct_assocs):
            write_json(f"struct_{i}.json", data)
            
        for i, data in enumerate(no_struct_assocs):
            write_json(f"no_struct_{i}.json", data)
            
        for i, data in enumerate(missing_fp):
            write_json(f"missing_fp_{i}.json", data)
            
        for i, data in enumerate(missing_binder):
            write_json(f"missing_binder_{i}.json", data)
            
        logger.info(
            f"Extraction summary: {len(struct_assocs)} struct, "
            f"{len(no_struct_assocs)} no_struct, "
            f"{len(missing_fp)} missing_fp, "
            f"{len(missing_binder)} missing_binder."
        )


# ---
# Main
# ---


def main() -> None:
    """Parse CLI arguments and execute extraction logic."""
    parser = argparse.ArgumentParser(
        description="Extract real LMDB test data into JSON format."
    )
    parser.add_argument(
        "--assocs-csv",
        type=Path,
        required=True,
        help="Path to the mapping CSV",
    )
    parser.add_argument(
        "--fingerprints-db",
        type=Path,
        required=True,
        help="Path to the target fingerprints LMDB",
    )
    parser.add_argument(
        "--binders-db",
        type=Path,
        required=True,
        help="Path to the binder features LMDB",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for test JSON fixtures",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable informative debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.assocs_csv.exists():
        logger.error(f"Associations CSV not found: {args.assocs_csv}")
        sys.exit(1)

    if not args.fingerprints_db.exists():
        logger.error(f"Fingerprints DB not found: {args.fingerprints_db}")
        sys.exit(1)
        
    if not args.binders_db.exists():
        logger.error(f"Binders DB not found: {args.binders_db}")
        sys.exit(1)

    extract_test_data(
        assocs_csv=args.assocs_csv,
        fingerprints_db=args.fingerprints_db,
        binders_db=args.binders_db,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

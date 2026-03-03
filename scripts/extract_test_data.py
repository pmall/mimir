"""
Script to extract real LMDB test data into JSON format for tokenizer/dataloader tests.

Produces a complete self-contained test dataset under tests/data/ including:
  - struct_0.json / struct_1.json       (real samples with binder structure)
  - no_struct_0.json / no_struct_1.json (real samples without binder structure)
  - missing_fp_0.json                   (assoc whose fingerprint is absent from LMDB)
  - missing_binder_0.json               (assoc whose binder is absent from LMDB)
  - associations.csv                    (all 6 rows, consumed by MimirDataset)
  - fingerprints.lmdb/                  (fingerprints for all but the missing-fp row)
  - binders.lmdb/                       (binders for all but the missing-binder row)

When no real example exists for missing_binder, a synthetic placeholder is created.

Usage:
    uv run python -m scripts.extract_test_data \\
        --config data/run78-v2/config.json \\
        -o tests/data [-v]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import lmdb
import msgpack

from mimir.config import load_config

# ---
# Constants
# ---

logger = logging.getLogger(__name__)

LMDB_MAP_SIZE = 1024 * 1024 * 1024  # 1 GB — plenty for a tiny test dataset

# Sentinel binder ID that will never appear in any real LMDB
_SYNTHETIC_MISSING_BINDER_ID = "__mimir_test_missing_binder__"

# CSV header ordered to match the real dataset
_CSV_COLUMNS = ["target", "binder_id"]


# ---
# Helpers
# ---


def _pack(obj: dict) -> bytes:
    """msgpack-encodes a dict the same way the production LMDB does."""
    return msgpack.packb(obj, use_bin_type=True)


def _make_synthetic_missing_binder_assoc(struct_assoc: dict) -> dict:
    """Build a synthetic missing-binder assoc from a real struct fixture.

    Takes the target from a real struct sample but uses the sentinel binder_id
    so that the binder lookup will always miss in the test LMDB.

    Args:
        struct_assoc: A real fixture dict with 'assoc' and 'fingerprint' keys.

    Returns:
        Dict with only 'assoc', where binder_id is the sentinel value.
    """
    assoc_row = dict(struct_assoc["assoc"])
    assoc_row["binder_id"] = _SYNTHETIC_MISSING_BINDER_ID
    return {"assoc": assoc_row}


# ---
# Feature extraction
# ---


def extract_test_data(
    assocs_csv: Path,
    fingerprints_db: Path,
    binders_db: Path,
    output_dir: Path,
) -> None:
    """Extracts test fixtures and builds a self-contained test dataset.

    Scans the associations CSV for real examples of each fixture type. For
    types not found in real data (e.g. no missing-binder case), a synthetic
    placeholder is created. Writes all JSON fixtures plus the associations CSV
    and the two LMDB databases consumed by MimirDataset / BucketBatchSampler.

    Args:
        assocs_csv: Path to the CSV mapping targets to binders.
        fingerprints_db: Path to the source fingerprints LMDB.
        binders_db: Path to the source binders LMDB.
        output_dir: Output directory for all test data artefacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening source LMDB environments...")
    fp_env = lmdb.open(str(fingerprints_db), readonly=True, lock=False)
    binders_env = lmdb.open(str(binders_db), readonly=True, lock=False)

    # --- Scan real data ---
    struct_assocs: list[dict] = []
    no_struct_assocs: list[dict] = []
    missing_fp: list[dict] = []
    missing_binder: list[dict] = []

    logger.info(f"Scanning associations from {assocs_csv}...")
    with fp_env.begin() as fp_txn, binders_env.begin() as bind_txn:
        with open(assocs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprot = row.get("target")
                binder_id = row.get("binder_id")

                fp_data = fp_txn.get(uniprot.encode("utf-8")) if uniprot else None
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

                if (
                    len(struct_assocs) == 2
                    and len(no_struct_assocs) == 2
                    and len(missing_fp) == 1
                    and len(missing_binder) == 1
                ):
                    break

    logger.info(
        f"Scan summary: {len(struct_assocs)} struct, "
        f"{len(no_struct_assocs)} no_struct, "
        f"{len(missing_fp)} missing_fp, "
        f"{len(missing_binder)} missing_binder."
    )

    # Synthesize missing-binder if not found in real data
    if not missing_binder:
        if struct_assocs:
            logger.warning(
                "No real missing-binder case found in dataset. "
                "Synthesizing a placeholder from a real struct assoc."
            )
            missing_binder.append(_make_synthetic_missing_binder_assoc(struct_assocs[0]))
        else:
            logger.error("Cannot synthesize missing_binder: no struct_assocs collected.")
            sys.exit(1)

    # --- Write JSON fixtures ---
    def write_json(name: str, data: dict) -> None:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=2)
        logger.info(f"Saved JSON: {path.name}")

    for i, data in enumerate(struct_assocs):
        write_json(f"struct_{i}.json", data)

    for i, data in enumerate(no_struct_assocs):
        write_json(f"no_struct_{i}.json", data)

    for i, data in enumerate(missing_fp):
        write_json(f"missing_fp_{i}.json", data)

    for i, data in enumerate(missing_binder):
        write_json(f"missing_binder_{i}.json", data)

    # --- Build the test LMDB and CSV ---
    # All valid associations: struct + no_struct (both have real FP + binder data)
    valid_assocs = struct_assocs + no_struct_assocs

    # Write the test fingerprints LMDB (excludes the missing-fp entry)
    fp_lmdb_path = output_dir / "fingerprints.lmdb"
    fp_lmdb_env = lmdb.open(str(fp_lmdb_path), map_size=LMDB_MAP_SIZE)
    with fp_lmdb_env.begin(write=True) as txn:
        for item in valid_assocs:
            key = item["assoc"]["target"].encode("utf-8")
            txn.put(key, _pack(item["fingerprint"]))
    fp_lmdb_env.close()
    logger.info(f"Wrote fingerprints.lmdb ({len(valid_assocs)} entries)")

    # Write the test binders LMDB (excludes the missing-binder entry)
    bin_lmdb_path = output_dir / "binders.lmdb"
    bin_lmdb_env = lmdb.open(str(bin_lmdb_path), map_size=LMDB_MAP_SIZE)
    with bin_lmdb_env.begin(write=True) as txn:
        for item in valid_assocs:
            key = item["assoc"]["binder_id"].encode("utf-8")
            txn.put(key, _pack(item["binder"]))
    bin_lmdb_env.close()
    logger.info(f"Wrote binders.lmdb ({len(valid_assocs)} entries)")

    # Write the test associations CSV
    # Ordering: valid samples first, then the two skip-case rows
    all_rows: list[dict] = []
    for item in valid_assocs:
        row = item["assoc"]
        all_rows.append({"target": row["target"], "binder_id": row["binder_id"]})

    # missing_fp row: the FP key is absent from fingerprints.lmdb
    mfp_row = missing_fp[0]["assoc"]
    all_rows.append({"target": mfp_row["target"], "binder_id": mfp_row["binder_id"]})

    # missing_binder row: the binder_id is absent from binders.lmdb
    mb_row = missing_binder[0]["assoc"]
    all_rows.append({"target": mb_row["target"], "binder_id": mb_row["binder_id"]})

    csv_path = output_dir / "associations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Wrote associations.csv ({len(all_rows)} rows)")

    fp_env.close()
    binders_env.close()

    logger.info(
        f"Done. tests/data/ contains: {len(struct_assocs)+len(no_struct_assocs)} valid samples, "
        f"1 missing-fp row, 1 missing-binder row."
    )


# ---
# Main
# ---


def main() -> None:
    """Parse CLI arguments and execute extraction logic."""
    parser = argparse.ArgumentParser(
        description="Extract real LMDB test data into JSON format.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json",
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

    config = load_config(args.config)

    if not config.binders_merged.exists():
        logger.error(f"Associations CSV not found: {config.binders_merged}")
        sys.exit(1)

    if not config.features_fingerprints.exists():
        logger.error(f"Fingerprints DB not found: {config.features_fingerprints}")
        sys.exit(1)

    if not config.features_binders.exists():
        logger.error(f"Binders DB not found: {config.features_binders}")
        sys.exit(1)

    extract_test_data(
        assocs_csv=config.binders_merged,
        fingerprints_db=config.features_fingerprints,
        binders_db=config.features_binders,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

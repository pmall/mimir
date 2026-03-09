"""
Export human target fingerprints data into a CSV for the Next.js viewer.

Extracts pLDDT, rSASA, smoothed rSASA, mask from the fingerprints features LMDB.

Usage:
    uv run python -m scripts.dataset.export_viewer_metadata \\
        --config data/run78-v2/config.json \\
        -o data/viewer_data.csv \\
        [-v]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import lmdb
import msgpack
from tqdm import tqdm

from mimir.config import load_config
from mimir.features import compute_rsasa, get_fingerprint_mask, get_smoothed_rsasa

logger = logging.getLogger(__name__)


def export_viewer_metadata(fingerprints_lmdb: Path, targets_lmdb: Path, output_csv: Path) -> None:
    """Read fingerprints LMDB, fetch full target data, and write CSV."""
    logger.info(f"Opening fingerprints LMDB: {fingerprints_lmdb}")
    logger.info(f"Opening targets LMDB:      {targets_lmdb}")

    f_env = lmdb.open(str(fingerprints_lmdb), readonly=True, lock=False, readahead=False, meminit=False)
    t_env = lmdb.open(str(targets_lmdb), readonly=True, lock=False, readahead=False, meminit=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total_read": 0, "exported": 0}

    fieldnames = [
        "target_id",
        "sequence",
        "mask",
        "rsasa",
        "smoothed_rsasa",
        "plddt",
        "rsasa_threshold",
    ]

    with f_env.begin() as f_txn, t_env.begin() as t_txn:
        total_entries = f_txn.stat()["entries"]
        pbar = tqdm(total=total_entries, desc="Exporting Metadata", unit="target", dynamic_ncols=True)

        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

            for _key, value in f_txn.cursor():
                stats["total_read"] += 1

                obj_f = msgpack.unpackb(value, raw=False)
                target_id = obj_f["id"]

                # Fetch full target data
                val_t = t_txn.get(target_id.encode("utf-8"))
                if val_t is None:
                    logger.warning(f"Target {target_id} not found in targets LMDB. Skipping.")
                    pbar.update(1)
                    continue

                obj_t = msgpack.unpackb(val_t, raw=False)

                sequence = obj_t["sequence"]
                sasa = obj_t["sasa"]
                plddt = obj_t["residue_plddt"]
                pos_ids = obj_f["position_ids"]
                threshold = obj_f.get("rsasa_threshold")

                # Reconstruct mask from position_ids (which are 1-indexed in the LMDB)
                mask = [False] * len(sequence)
                for pid in pos_ids:
                    if 1 <= pid <= len(sequence):
                        mask[pid - 1] = True

                rsasa = compute_rsasa(sequence, sasa)
                smoothed_rsasa = get_smoothed_rsasa(rsasa)

                writer.writerow({
                    "target_id": target_id,
                    "sequence": sequence,
                    "mask": json.dumps(mask),
                    "rsasa": json.dumps([round(float(x), 3) for x in rsasa]),
                    "smoothed_rsasa": json.dumps([round(float(x), 3) for x in smoothed_rsasa]),
                    "plddt": json.dumps([round(float(x), 2) for x in plddt]),
                    "rsasa_threshold": threshold if threshold is not None else "",
                })

                stats["exported"] += 1
                pbar.update(1)

        pbar.close()

    f_env.close()
    t_env.close()

    logger.info("Processing complete.")
    logger.info(f"  Total fingerprints read: {stats['total_read']}")
    logger.info(f"  Exported to CSV:         {stats['exported']}")
    logger.info(f"  Output path:             {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Fingerprints LMDB to viewer CSV.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to the output csv file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log progress and statistics")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config)

    if not config.features_fingerprints.exists():
        logger.error(f"Fingerprints LMDB not found: {config.features_fingerprints}")
        sys.exit(1)
    if not config.features_targets.exists():
        logger.error(f"Targets LMDB not found: {config.features_targets}")
        sys.exit(1)

    export_viewer_metadata(
        fingerprints_lmdb=config.features_fingerprints,
        targets_lmdb=config.features_targets,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

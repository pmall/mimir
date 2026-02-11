"""
Extract raw structure features (Sequence, Coordinates, SASA) for ESM-3.

Reads a binder list CSV, looks up structures in an LMDB, and computes features
using ESM-3's ProteinChain.

Structure features are extracted only for PDB entries (type == "PDB"). For other
entry types (HH, VH), only the sequence is retained without structural data.

Output LMDB entry schema (msgpack-serialized dict):
    {
        "id":          str,                        # structure_id from CSV
        "target":      str,                        # target UniProt accession
        "sequence":    str,                        # amino-acid sequence (1-letter codes, length L)
        "coordinates": list[list[list[float]]],    # shape (L, 37, 3) — atom37 xyz in Å, or None
        "sasa":        list[float],                # per-residue SASA in Å², length L, or None
    }
"""

import argparse
import csv
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from mimir.structure_features import StructureFeatures, parse_mmcif_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-worker LMDB handle
# ---------------------------------------------------------------------------
_STRUCT_ENV: lmdb.Environment | None = None


def _init_worker(structures_db_path: str) -> None:
    """Initializer called once per worker process to open a read-only LMDB handle."""
    global _STRUCT_ENV
    _STRUCT_ENV = lmdb.open(
        structures_db_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=256,
    )


def compute_features(
    entry_id: str,
    sequence: str,
    entry_type: str,
    target: str,
    has_structure: bool,
) -> tuple[dict[str, Any], str | None]:
    """Compute structure features for a single entry.

    For PDB entries, extracts coordinates and SASA from the structure.
    For non-PDB entries (HH, VH), returns only the sequence with None for
    structural features.

    Args:
        entry_id: The structure_id (used as LMDB key).
        sequence: The amino-acid sequence from the CSV.
        entry_type: Entry type label (PDB, HH, VH).
        target: Target UniProt accession from the CSV.
        has_structure: Whether a structure exists in the input LMDB.

    Returns:
        Tuple of (features_dict, error_message_or_None).
    """
    features = StructureFeatures(
        entry_id=entry_id,
        target=target,
        sequence=sequence,
        coordinates=None,
        sasa=None,
    )

    # Non-PDB entries (HH, VH) don't have structures - return sequence only
    if not has_structure:
        if entry_type == "PDB":
            raise ValueError(f"Structure missing for PDB entry {entry_id}")
        return features.to_dict(), None

    # Fetch compressed mmCIF from the worker's own LMDB handle
    assert _STRUCT_ENV is not None
    with _STRUCT_ENV.begin() as txn:
        structure_content = txn.get(entry_id.encode("utf-8"))

    if structure_content is None:
        if entry_type == "PDB":
            raise ValueError(f"Structure missing for PDB entry {entry_id}")
        return features.to_dict(), None

    try:
        # Parse mmCIF and extract 3-track features
        struct_seq, coords, sasa = parse_mmcif_bytes(structure_content, compressed=True)

        # Update features with extracted data
        features.sequence = struct_seq
        features.coordinates = coords
        features.sasa = sasa

    except Exception as e:
        error_msg = str(e)
        # Handle known ProteinChain issues gracefully
        if "residue names exceed" in error_msg or "non-standard" in error_msg.lower():
            logger.warning(f"Skipping {entry_id}: {error_msg}")
            return features.to_dict(), f"Non-standard residues: {error_msg}"
        raise ValueError(f"Feature extraction failed for entry {entry_id}: {e}") from e

    return features.to_dict(), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structure features for ESM-3")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/run78-v2/binders_lists/final_binders_96aa.csv"),
        help="Path to the binder list CSV",
    )
    parser.add_argument(
        "--structures-db",
        type=Path,
        default=Path("data/run78-v2/binders_structures"),
        help="Path to the LMDB containing zstd-compressed mmCIFs",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("data/run78-v2/binders_features"),
        help="Path to the output LMDB for features",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, multiprocessing.cpu_count()),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for LMDB writes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of entries to process (for testing)",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        logger.error(f"Input CSV not found: {args.input_csv}")
        sys.exit(1)

    if not args.structures_db.exists():
        logger.error(f"Structures DB not found: {args.structures_db}")
        sys.exit(1)

    # Read CSV
    logger.info(f"Reading CSV: {args.input_csv}")
    entries_to_process: list[tuple[str, str, str, str]] = []

    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_id = row.get("structure_id")
            sequence = row.get("sequence")
            entry_type = row.get("type", "UNKNOWN")
            target = row.get("target", "UNKNOWN")

            if not entry_id or not sequence:
                continue

            entries_to_process.append((entry_id, sequence, entry_type, target))

            if args.limit and len(entries_to_process) >= args.limit:
                break

    logger.info(f"Found {len(entries_to_process)} entries to process.")

    # Pre-check structure availability - only for PDB entries
    logger.info("Checking structure availability in LMDB...")
    tasks: list[tuple[str, str, str, str, bool]] = []

    pdb_entries = [e for e in entries_to_process if e[2] == "PDB"]
    non_pdb_count = len(entries_to_process) - len(pdb_entries)

    if pdb_entries:
        struct_env = lmdb.open(
            str(args.structures_db),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=256,
        )
        with struct_env.begin() as txn:
            for entry_id, sequence, entry_type, target in tqdm(
                entries_to_process, desc="Checking structures"
            ):
                # Only check LMDB for PDB entries - HH/VH never have structures
                has_structure = entry_type == "PDB" and txn.get(
                    entry_id.encode("utf-8")
                ) is not None
                tasks.append((entry_id, sequence, entry_type, target, has_structure))
        struct_env.close()
    else:
        # No PDB entries - skip LMDB entirely
        for entry_id, sequence, entry_type, target in entries_to_process:
            tasks.append((entry_id, sequence, entry_type, target, False))

    pdb_count = len(pdb_entries)
    structures_found = sum(1 for *_, h in tasks if h)
    logger.info(
        f"PDB entries: {pdb_count}, Non-PDB: {non_pdb_count}, "
        f"Structures found: {structures_found}/{pdb_count}"
    )

    # Prepare output LMDB (clear existing for idempotency)
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    if args.output_db.exists():
        logger.warning(f"Clearing existing output LMDB: {args.output_db}")
        import shutil
        shutil.rmtree(args.output_db)
    map_size = 10 * 1024 * 1024 * 1024  # 10 GB
    output_env = lmdb.open(str(args.output_db), map_size=map_size, writemap=True)

    # Process entries in chunks to control memory usage
    logger.info(f"Starting feature extraction with {args.num_workers} workers...")

    chunk_size = 500  # Process 500 entries at a time to limit memory
    stats = {
        "total_input": len(tasks),
        "pdb_with_structure": 0,
        "pdb_sequence_only": 0,
        "pdb_critical_error": 0,
        "hh_vh_sequence_only": 0,
        "total_saved": 0,
    }
    pbar = tqdm(total=len(tasks), desc="Extracting features", unit="entry")

    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i : i + chunk_size]

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(str(args.structures_db),),
        ) as executor:
            futures = {
                executor.submit(
                    compute_features, entry_id, sequence, entry_type, target, has_struct
                ): (entry_id, entry_type)
                for entry_id, sequence, entry_type, target, has_struct in chunk
            }

            for future in as_completed(futures):
                entry_id, entry_type = futures[future]
                saved = False
                try:
                    features, error = future.result(timeout=300)  # 5 min timeout
                    saved = True
                except TimeoutError:
                    logger.error(f"CRITICAL: Timeout - {entry_id} ({entry_type}) NOT SAVED")
                    if entry_type == "PDB":
                        stats["pdb_critical_error"] += 1
                    continue
                except Exception as e:
                    logger.error(f"CRITICAL: Exception - {entry_id} ({entry_type}) NOT SAVED: {e}")
                    if entry_type == "PDB":
                        stats["pdb_critical_error"] += 1
                    continue

                if saved:
                    has_coords = features["coordinates"] is not None
                    
                    if entry_type == "PDB":
                        if has_coords:
                            stats["pdb_with_structure"] += 1
                        else:
                            stats["pdb_sequence_only"] += 1
                            if error:
                                logger.warning(f"PDB {entry_id}: Structure parsing failed, saved sequence-only: {error}")
                    else:
                        # HH/VH entries
                        stats["hh_vh_sequence_only"] += 1

                    # Write immediately to avoid memory buildup
                    with output_env.begin(write=True) as txn:
                        txn.put(entry_id.encode("utf-8"), msgpack.packb(features))
                    stats["total_saved"] += 1

                pbar.update(1)

    pbar.close()
    output_env.close()

    logger.info("Processing complete.")
    logger.info(f"  Total input entries:       {stats['total_input']}")
    logger.info("")
    logger.info(f"  PDB entries:")
    logger.info(f"    With structure (3 tracks): {stats['pdb_with_structure']}")
    logger.info(f"    Sequence-only (recovered): {stats['pdb_sequence_only']}")
    logger.info(f"    Critical errors (not saved): {stats['pdb_critical_error']}")
    logger.info("")
    logger.info(f"  Non-PDB entries (HH/VH):")
    logger.info(f"    Sequence-only:             {stats['hh_vh_sequence_only']}")
    logger.info("")
    logger.info(f"  Total saved:               {stats['total_saved']}")
    logger.info(f"  Success rate:              {stats['total_saved']}/{stats['total_input']} ({100*stats['total_saved']/stats['total_input']:.2f}%)")


if __name__ == "__main__":
    main()

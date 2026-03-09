"""
Extract raw structure features (Sequence, Coordinates) for ESM-3.

Reads a binder list CSV, looks up structures in an LMDB, and computes features
using ESM-3's ProteinChain.

Structure features are extracted only for PDB entries (type == "PDB"). For other
entry types (HH, VH), only the sequence is retained without structural data.

Output LMDB entry schema (msgpack-serialized dict):
    {
        "id":               str,        # binder_id from CSV
        "sequence":         str,        # amino-acid sequence (1-letter codes, length L)
        "structure_tokens": list[int],  # ESM-3 structure integer tokens (1D list), or None
    }

Usage:
    uv run python -m scripts.dataset.extract_binders_features \\
        --config data/run78-v2/config.json \\
        [--num-workers 4] [--chunk-size 500] [--limit N] [-v]
"""

import argparse
import csv
import logging
import multiprocessing
import shutil
import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
import torch
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.utils import encoding
from tqdm import tqdm

from mimir.config import load_config

from mimir.features import BinderFeatures, parse_binder_mmcif_bytes

# ---
# Constants
# ---

LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB virtual (sparse on Linux, no pre-alloc)


warnings.filterwarnings("ignore", category=UserWarning, module="esm.models.vqvae")
warnings.filterwarnings("ignore", category=FutureWarning, module="esm.models.vqvae")

logger = logging.getLogger(__name__)


# ---
# Per-worker globals
# ---

_STRUCT_ENV: lmdb.Environment | None = None
_STRUCTURE_ENCODER = None
_STRUCTURE_TOKENIZER = None


def _init_worker(structures_db_path: str) -> None:
    """Initialize per-worker LMDB handle and ESM-3 structure encoder.

    Called once per worker process by ProcessPoolExecutor.

    Args:
        structures_db_path: Path to the read-only structures LMDB.
    """
    global _STRUCT_ENV, _STRUCTURE_ENCODER, _STRUCTURE_TOKENIZER
    torch.set_num_threads(1)
    _STRUCT_ENV = lmdb.open(
        structures_db_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=256,
    )
    # Load on CPU to avoid GPU OOM with multiprocessing
    _STRUCTURE_ENCODER = ESM3_structure_encoder_v0(torch.device("cpu")).eval()
    _STRUCTURE_TOKENIZER = StructureTokenizer()


# ---
# Feature extraction
# ---


def compute_features(
    entry_id: str,
    binder_id: str,
    sequence: str,
    entry_type: str,
    has_structure: bool,
) -> tuple[dict[str, Any], str | None]:
    """Compute structure features for a single entry.

    For PDB entries, extracts structure tokens from the mmCIF.
    For non-PDB entries (HH, VH), returns only the sequence with None for
    structural features.

    Args:
        entry_id: The structure_id used to look up the mmCIF in the LMDB.
        binder_id: The unique binder key (written as the LMDB output key).
        sequence: The amino-acid sequence from the CSV.
        entry_type: Entry type label (PDB, HH, VH).
        has_structure: Whether a structure is present in the input LMDB.

    Returns:
        Tuple of (features_dict, error_message_or_None).
    """
    features = BinderFeatures(
        entry_id=binder_id,
        sequence=sequence,
        structure_tokens=None,
    )

    # Non-PDB entries (HH, VH) have no structure — return sequence only
    if not has_structure:
        if entry_type == "PDB":
            raise ValueError(f"Structure missing for PDB entry {entry_id}")
        return features.to_dict(), None

    # Fetch compressed mmCIF from the worker-local LMDB handle
    assert _STRUCT_ENV is not None
    with _STRUCT_ENV.begin() as txn:
        structure_content = txn.get(entry_id.encode("utf-8"))

    if structure_content is None:
        if entry_type == "PDB":
            raise ValueError(f"Structure not found in LMDB for PDB entry {entry_id}")
        return features.to_dict(), None

    try:
        # For PDB entries binder_id is "<PDB_ID>_<chain>", e.g. "1BH9_A"
        chain_id = None
        if entry_type == "PDB" and "_" in binder_id:
            chain_id = binder_id.split("_")[-1]

        # Parse mmCIF and extract aligned features for the specific chain
        parsed_structure = parse_binder_mmcif_bytes(
            structure_content, reference_sequence=sequence, compressed=True, chain_id=chain_id
        )

        if parsed_structure is None:
            # Full void
            return features.to_dict(), None

        # Tokenize structure coordinates into discrete ESM-3 tokens
        with torch.no_grad():
            _, _, struct_tokens = encoding.tokenize_structure(
                coordinates=torch.tensor(parsed_structure.coords, dtype=torch.float32),
                structure_encoder=_STRUCTURE_ENCODER,
                structure_tokenizer=_STRUCTURE_TOKENIZER,
                reference_sequence=parsed_structure.sequence,
                add_special_tokens=False,
            )

        features.sequence = parsed_structure.sequence
        features.structure_tokens = struct_tokens.flatten().tolist()

    except Exception as e:
        # Degrade to sequence-only so the entry is never lost
        error_msg = str(e)
        logger.error(
            f"PDB {entry_id}: Structure extraction failed, saving sequence-only: {error_msg}"
        )
        return features.to_dict(), error_msg

    return features.to_dict(), None


# ---
# Main
# ---


def main() -> None:
    """Parse CLI arguments and initiate binder feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract structure features for ESM-3"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, multiprocessing.cpu_count()),
        help="Number of worker processes (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of entries per worker pool batch (default: 500)",
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

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config)

    if not config.binders_merged.exists():
        logger.error(f"Input CSV not found: {config.binders_merged}")
        sys.exit(1)

    if not config.structures_pdb.exists():
        logger.error(f"Structures DB not found: {config.structures_pdb}")
        sys.exit(1)

    extract_binders_features(
        input_csv=config.binders_merged,
        structures_db=config.structures_pdb,
        output=config.features_binders,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )

def extract_binders_features(
    input_csv: Path,
    structures_db: Path,
    output: Path,
    num_workers: int = 4,
    chunk_size: int = 500,
    limit: int | None = None,
) -> None:
    """Extract ESM-3 features sequentially from an LMDB database based on a CSV list.
    
    Args:
        input_csv: Path to the binders CSV containing IDs and sequences.
        structures_db: Path to the LMDB containing compressed mmCIFs.
        output: Path to the output LMDB where features will be written.
        num_workers: Concurrent workers for rendering the features.
        chunk_size: Inner batch handling.
        limit: Max entries to limit the run to.
    """
    # ---
    # Read CSV
    # ---
    logger.info(f"Reading CSV: {input_csv}")
    entries_to_process: list[tuple[str, str, str, str]] = []

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_id = row.get("structure_id", "")
            binder_id = row.get("binder_id", "")
            sequence = row.get("sequence", "")
            entry_type = row.get("type", "UNKNOWN")

            if not sequence or not binder_id:
                continue

            entries_to_process.append((entry_id, binder_id, sequence, entry_type))

            if limit and len(entries_to_process) >= limit:
                break

    logger.info(f"Found {len(entries_to_process)} entries to process.")

    # ---
    # Check structure availability in LMDB
    # ---
    logger.info("Checking structure availability in LMDB...")
    tasks: list[tuple[str, str, str, str, bool]] = []

    pdb_entries = [e for e in entries_to_process if e[3] == "PDB"]
    non_pdb_count = len(entries_to_process) - len(pdb_entries)

    if pdb_entries:
        struct_env = lmdb.open(
            str(structures_db),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=256,
        )
        with struct_env.begin() as txn:
            for entry_id, binder_id, sequence, entry_type in tqdm(
                entries_to_process, desc="Checking structures"
            ):
                # Only check LMDB for PDB entries — HH/VH never have structures
                has_structure = entry_type == "PDB" and txn.get(
                    entry_id.encode("utf-8")
                ) is not None
                tasks.append((entry_id, binder_id, sequence, entry_type, has_structure))
        struct_env.close()
    else:
        # No PDB entries — skip LMDB entirely
        for entry_id, binder_id, sequence, entry_type in entries_to_process:
            tasks.append((entry_id, binder_id, sequence, entry_type, False))

    pdb_count = len(pdb_entries)
    structures_found = sum(1 for *_, h in tasks if h)
    logger.info(
        f"PDB entries: {pdb_count}, Non-PDB: {non_pdb_count}, "
        f"Structures found: {structures_found}/{pdb_count}"
    )

    # ---
    # Prepare output LMDB
    # ---
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        logger.warning(f"Clearing existing output LMDB: {output}")
        shutil.rmtree(output)

    output_env = lmdb.open(str(output), map_size=LMDB_MAP_SIZE)

    # ---
    # Feature extraction
    # ---
    logger.info(f"Starting feature extraction with {num_workers} workers...")

    stats = {
        "total_input": len(tasks),
        "pdb_with_structure": 0,
        "pdb_sequence_only": 0,
        "pdb_critical_error": 0,
        "hh_vh_sequence_only": 0,
        "total_saved": 0,
    }
    pbar = tqdm(
        total=len(tasks),
        desc="Extracting features",
        unit="entry",
        miniters=1,
        smoothing=0.1,
        dynamic_ncols=True,
        leave=True,
    )

    ctx = multiprocessing.get_context("spawn")
    
    # Bounded futures to prevent memory explosion from fast reading
    max_pending = num_workers * 2
    active_futures = {}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(structures_db),),
        max_tasks_per_child=50,
    ) as executor:
        for entry_id, binder_id, sequence, entry_type, has_struct in tasks:
            future = executor.submit(
                compute_features, entry_id, binder_id, sequence, entry_type, has_struct
            )
            active_futures[future] = (entry_id, binder_id, entry_type)
            
            # If we've hit our pending tasks limit, wait for at least one to finish
            while len(active_futures) >= max_pending:
                done, _ = wait(
                    active_futures.keys(),
                    return_when=FIRST_COMPLETED
                )
                for done_future in done:
                    d_entry_id, d_binder_id, d_entry_type = active_futures.pop(done_future)
                    saved = False
                    try:
                        features, error = done_future.result(timeout=300)  # 5 min timeout
                        saved = True
                    except TimeoutError:
                        logger.error(
                            f"CRITICAL: Timeout - {d_entry_id} ({d_entry_type}) NOT SAVED"
                        )
                        if d_entry_type == "PDB":
                            stats["pdb_critical_error"] += 1
                        pbar.update(1)
                        continue
                    except Exception as e:
                        logger.error(
                            f"CRITICAL: Exception - {d_entry_id} ({d_entry_type}) NOT SAVED: {e}"
                        )
                        if d_entry_type == "PDB":
                            stats["pdb_critical_error"] += 1
                        pbar.update(1)
                        continue

                    if saved:
                        has_coords = features["structure_tokens"] is not None

                        if d_entry_type == "PDB":
                            if has_coords:
                                stats["pdb_with_structure"] += 1
                            else:
                                stats["pdb_sequence_only"] += 1
                        else:
                            stats["hh_vh_sequence_only"] += 1

                        # Write immediately to avoid memory buildup
                        with output_env.begin(write=True) as txn:
                            txn.put(d_binder_id.encode("utf-8"), msgpack.packb(features))
                        stats["total_saved"] += 1

                    pbar.update(1)

        # Drain the remaining futures
        for done_future in as_completed(active_futures.keys()):
            d_entry_id, d_binder_id, d_entry_type = active_futures[done_future]
            saved = False
            try:
                features, error = done_future.result(timeout=300)
                saved = True
            except TimeoutError:
                logger.error(
                    f"CRITICAL: Timeout - {d_entry_id} ({d_entry_type}) NOT SAVED"
                )
                if d_entry_type == "PDB":
                    stats["pdb_critical_error"] += 1
                pbar.update(1)
                continue
            except Exception as e:
                logger.error(
                    f"CRITICAL: Exception - {d_entry_id} ({d_entry_type}) NOT SAVED: {e}"
                )
                if d_entry_type == "PDB":
                    stats["pdb_critical_error"] += 1
                pbar.update(1)
                continue

            if saved:
                has_coords = features["structure_tokens"] is not None

                if d_entry_type == "PDB":
                    if has_coords:
                        stats["pdb_with_structure"] += 1
                    else:
                        stats["pdb_sequence_only"] += 1
                else:
                    stats["hh_vh_sequence_only"] += 1

                with output_env.begin(write=True) as txn:
                    txn.put(d_binder_id.encode("utf-8"), msgpack.packb(features))
                stats["total_saved"] += 1

            pbar.update(1)

    pbar.close()
    output_env.close()

    # ---
    # Summary
    # ---
    logger.info("Processing complete.")
    logger.info(f"  Total input:                 {stats['total_input']}")
    logger.info("")
    logger.info("  PDB entries:")
    logger.info(f"    With structure (3 tracks):  {stats['pdb_with_structure']}")
    logger.info(f"    Sequence-only (degraded):   {stats['pdb_sequence_only']}")
    logger.info(f"    Critical errors (not saved): {stats['pdb_critical_error']}")
    logger.info("")
    logger.info("  Non-PDB entries (HH/VH):")
    logger.info(f"    Sequence-only:              {stats['hh_vh_sequence_only']}")
    logger.info("")
    logger.info(f"  Total saved:                 {stats['total_saved']}")
    logger.info(
        f"  Success rate:                {stats['total_saved']}/{stats['total_input']} "
        f"({100 * stats['total_saved'] / stats['total_input']:.2f}%)"
    )


if __name__ == "__main__":
    main()

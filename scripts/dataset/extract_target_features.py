"""
Extract human protein target features from an AlphaFold2 bulk download tarball.

Filters to retain only valid, single-fragment F1 UniProt sequences.
Extracts the global pLDDT metric, standard sequence, structural tokens (3 tracks), and SASA.
Output is msgpack-serialized to an LMDB using the FingerprintFeatures schema.

Usage:
    uv run python -m scripts.dataset.extract_target_features \
        --tar-file data/UP000005640_9606_HUMAN_v6.tar \
        -o data/run78-v2/features_targets \
        [--num-workers 4] [--chunk-size 500] [--verbose]
"""

import argparse
import gzip
import logging
import multiprocessing
import re
import shutil
import sys
import tarfile
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning, module="esm.models.vqvae")
warnings.filterwarnings("ignore", category=FutureWarning, module="esm.models.vqvae")

import lmdb
import msgpack
import torch
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.utils import encoding
from tqdm import tqdm

from mimir.features import TargetFeatures, parse_af2_mmcif_bytes

# ---
# Constants
# ---

LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB virtual (sparse on Linux, no pre-alloc)


logger = logging.getLogger(__name__)


# ---
# Per-worker globals
# ---

_STRUCTURE_ENCODER = None
_STRUCTURE_TOKENIZER = None


def _init_worker() -> None:
    """Initialize per-worker ESM-3 structure encoder.

    Called once per worker process by ProcessPoolExecutor.
    """
    global _STRUCTURE_ENCODER, _STRUCTURE_TOKENIZER
    # Load on CPU to avoid GPU OOM with multiprocessing
    import torch
    torch.set_num_threads(1)  # Prevent OpenMP thread contention during multiprocessing
    
    _STRUCTURE_ENCODER = ESM3_structure_encoder_v0(torch.device("cpu")).eval()
    _STRUCTURE_TOKENIZER = StructureTokenizer()


# ---
# Feature extraction
# ---


def compute_features(
    entry_id: str,
    cif_bytes: bytes,
) -> tuple[dict[str, Any] | None, str | None]:
    """Compute target features for a single AF2 mmCIF sequence.

    Args:
        entry_id: The UniProt accession.
        cif_bytes: Decompressed mmCIF file bytes.

    Returns:
        Tuple of (features_dict, error_message_or_None).
        If features_dict is None, parsing failed entirely.
    """
    try:
        parsed_target = parse_af2_mmcif_bytes(cif_bytes, compressed=False)
        
        # Tokenize structure coordinates into discrete ESM-3 tokens
        assert _STRUCTURE_ENCODER is not None
        assert _STRUCTURE_TOKENIZER is not None

        with torch.no_grad():
            _, _, struct_tokens = encoding.tokenize_structure(
                coordinates=torch.tensor(parsed_target.coords, dtype=torch.float32),
                structure_encoder=_STRUCTURE_ENCODER,
                structure_tokenizer=_STRUCTURE_TOKENIZER,
                reference_sequence=parsed_target.sequence,
                add_special_tokens=False,
            )

        features = TargetFeatures(
            entry_id=entry_id,
            sequence=parsed_target.sequence,
            structure_tokens=struct_tokens.squeeze(0).tolist(),
            sasa=parsed_target.sasa.tolist(),
            plddt=parsed_target.global_plddt,
            residue_plddt=parsed_target.residue_plddt.tolist(),
            position_ids=list(range(1, len(parsed_target.sequence) + 1)),
        )
        
        return features.to_dict(), None

    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Target {entry_id}: Structure extraction failed: {error_msg}")
        return None, error_msg

# ---
# Main
# ---


def main() -> None:
    """Parse CLI arguments and initiate target feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract target features for ESM-3 from AF2 tarball"
    )
    parser.add_argument(
        "--tar-file",
        type=Path,
        required=True,
        help="Path to the AF2 tarball file (e.g., UP000005640_9606_HUMAN_v6.tar)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to the output LMDB for target features",
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

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.tar_file.exists():
        logger.error(f"Tar file not found: {args.tar_file}")
        sys.exit(1)

    extract_target_features(
        tar_file=args.tar_file,
        output=args.output,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )

def extract_target_features(
    tar_file: Path,
    output: Path,
    num_workers: int = 4,
    chunk_size: int = 500,
    limit: int | None = None,
) -> None:
    """Extract ESM-3 features from a bulk AlphaFold2 mmCIF tarball.

    Args:
        tar_file: Path to the AlphaFold2 .tar source file.
        output: Path to the output LMDB directory.
        num_workers: Number of concurrent worker processes.
        chunk_size: Processing batch size.
        limit: Optional maximum number of targets to process.
    """
    # ---
    # First Pass: Find Valid UniProts
    # ---
    logger.info(f"Scanning Tarball: {tar_file}")
    
    # regex matches: AF-A0A024R1R8-F1-model_v6.cif.gz -> group 1 = A0A024R1R8, group 2 = 1
    pattern = re.compile(r"AF-([A-Z0-9]+)-F(\d+)-model_v\d+\.cif\.gz")
    
    uniprot_fragments: dict[str, set[int]] = {}
    
    with tarfile.open(tar_file, "r|") as tar:
        for item in tar:
            match = pattern.search(item.name)
            if match:
                uniprot = match.group(1)
                frag = int(match.group(2))
                if uniprot not in uniprot_fragments:
                    uniprot_fragments[uniprot] = set()
                uniprot_fragments[uniprot].add(frag)
                
    multi_fragment_uniprots = {k for k, v in uniprot_fragments.items() if len(v) > 1}
    valid_uniprots = {k for k, v in uniprot_fragments.items() if len(v) == 1 and 1 in v}
    
    logger.info(f"Unique UniProt accessions found: {len(uniprot_fragments)}")
    logger.info(f"Multi-fragment (discarded):      {len(multi_fragment_uniprots)}")
    logger.info(f"Valid single-fragment F1:        {len(valid_uniprots)}")

    if limit:
        valid_uniprots = set(list(valid_uniprots)[:limit])
        logger.info(f"Limited run to {len(valid_uniprots)} targets.")

    # ---
    # Prepare output LMDB
    # ---
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        logger.warning(f"Clearing existing output LMDB: {output}")
        shutil.rmtree(output)

    output_env = lmdb.open(str(output), map_size=LMDB_MAP_SIZE)

    # ---
    # Feature extraction (Second Pass)
    # ---
    logger.info(f"Starting feature extraction with {num_workers} workers...")

    stats = {
        "total_attempted": 0,
        "saved_successfully": 0,
        "failed_extraction": 0,
    }

    pbar = tqdm(
        total=len(valid_uniprots),
        desc="Extracting features",
        unit="target",
        miniters=1,
        smoothing=0.1,
        dynamic_ncols=True,
        leave=True,
    )

    ctx = multiprocessing.get_context("spawn")
    
    # Bounded futures to prevent memory explosion from fast tarball reading
    max_pending = num_workers * 2
    active_futures = {}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        max_tasks_per_child=50,
    ) as executor:
        
        with tarfile.open(tar_file, "r|") as tar:
            for item in tar:
                if not item.isfile():
                    continue
                match = pattern.search(item.name)
                if not match:
                    continue
                
                uniprot = match.group(1)
                if uniprot not in valid_uniprots:
                    continue
                
                f = tar.extractfile(item)
                if f is None:
                    continue
                
                cif_compressed_bytes = f.read()
                try:
                    cif_bytes = gzip.decompress(cif_compressed_bytes)
                except Exception as e:
                    logger.error(f"Failed to decompress gzip for {uniprot}: {e}")
                    stats["failed_extraction"] += 1
                    pbar.update(1)
                    continue

                stats["total_attempted"] += 1
                
                future = executor.submit(compute_features, uniprot, cif_bytes)
                active_futures[future] = uniprot
                
                # If we've hit our pending tasks limit, wait for at least one to finish
                while len(active_futures) >= max_pending:
                    done, _ = wait(
                        active_futures.keys(),
                        return_when=FIRST_COMPLETED
                    )
                    for done_future in done:
                        e_id = active_futures.pop(done_future)
                        try:
                            features_dict, err = done_future.result(timeout=300)
                            if features_dict:
                                with output_env.begin(write=True) as txn:
                                    txn.put(e_id.encode("utf-8"), msgpack.packb(features_dict))
                                stats["saved_successfully"] += 1
                            else:
                                stats["failed_extraction"] += 1
                        except TimeoutError:
                            logger.error(f"CRITICAL: Timeout - {e_id} NOT SAVED")
                            stats["failed_extraction"] += 1
                        except Exception as e:
                            logger.error(f"CRITICAL: Exception - {e_id} NOT SAVED: {e}")
                            stats["failed_extraction"] += 1
                        pbar.update(1)

        # Drain the remaining futures after the tarball is fully read
        for done_future in as_completed(active_futures.keys()):
            e_id = active_futures[done_future]
            try:
                features_dict, err = done_future.result(timeout=300)
                if features_dict:
                    with output_env.begin(write=True) as txn:
                        txn.put(e_id.encode("utf-8"), msgpack.packb(features_dict))
                    stats["saved_successfully"] += 1
                else:
                    stats["failed_extraction"] += 1
            except TimeoutError:
                logger.error(f"CRITICAL: Timeout - {e_id} NOT SAVED")
                stats["failed_extraction"] += 1
            except Exception as e:
                logger.error(f"CRITICAL: Exception - {e_id} NOT SAVED: {e}")
                stats["failed_extraction"] += 1
            pbar.update(1)

    pbar.close()
    output_env.close()

    # ---
    # Summary
    # ---
    logger.info("Processing complete.")
    logger.info(f"  Valid targets identified:    {len(valid_uniprots)}")
    logger.info(f"  Total targets attempted:     {stats['total_attempted']}")
    logger.info(f"  Saved successfully:          {stats['saved_successfully']}")
    logger.info(f"  Failed extraction:           {stats['failed_extraction']}")
    logger.info("")
    if stats['total_attempted'] > 0:
        logger.info(
            f"  Success rate:                {stats['saved_successfully']}/{stats['total_attempted']} "
            f"({100 * stats['saved_successfully'] / stats['total_attempted']:.2f}%)"
        )


if __name__ == "__main__":
    main()

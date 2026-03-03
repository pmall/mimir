"""
Download PDB Structures into LMDB

Downloads mmCIF structure files from the RCSB PDB for all PDB-sourced binders
in a binders CSV file. Structures are stored in an LMDB database with zstandard
compression. Each entry is keyed by its PDB ID (e.g. b"1A09") and the value is
the zstd-compressed CIF content.

The LMDB acts as a resumable cache: entries already present are skipped, so
partial downloads can be resumed without re-fetching.

Usage:
    uv run python -m scripts.dataset.download_pdb_structures \\
        --config data/run78-v2/config.json \\
        [--max N] [--concurrency 10] [-v]
"""

import argparse
import asyncio
import csv
import gzip
import logging
import sys
from pathlib import Path

import httpx
import lmdb
import zstandard as zstd

from mimir.config import load_config

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---
# Constants
# ---

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
DEFAULT_CONCURRENCY = 10
LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB virtual (sparse on Linux, no pre-alloc)
ZSTD_LEVEL = 3


# ---
# Data collection
# ---


def collect_pdb_ids(binders_path: Path) -> list[str]:
    """Extract unique PDB IDs from a binders CSV (type == PDB rows).

    Args:
        binders_path: Path to the binders CSV file.

    Returns:
        Sorted list of unique PDB IDs.
    """
    ids: set[str] = set()
    with open(binders_path) as f:
        for row in csv.DictReader(f):
            if row["type"] == "PDB":
                ids.add(row["structure_id"].upper())
    return sorted(ids)


def get_existing_ids(env: lmdb.Environment) -> set[str]:
    """Return the set of PDB IDs already stored in the LMDB.

    Args:
        env: An open LMDB environment.

    Returns:
        Set of PDB ID strings present in the database.
    """
    existing: set[str] = set()
    with env.begin() as txn:
        cursor = txn.cursor()
        for key in cursor.iternext(values=False):
            existing.add(key.decode("utf-8"))
    return existing


# ---
# Download
# ---


async def download_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    pdb_id: str,
) -> tuple[str, bytes | None, str]:
    """Download a single mmCIF file from RCSB.

    Args:
        client: An httpx async client.
        semaphore: Concurrency limiter.
        pdb_id: The PDB identifier to download.

    Returns:
        Tuple of (pdb_id, decompressed_content_or_None, status_message).
    """
    url = RCSB_CIF_URL.format(pdb_id=pdb_id)
    async with semaphore:
        try:
            resp = await client.get(url, timeout=60.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            return pdb_id, None, f"HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return pdb_id, None, str(e)

    try:
        content = gzip.decompress(resp.content)
    except Exception:
        content = resp.content

    return pdb_id, content, "downloaded"


# ---
# Verification
# ---


def verify_lmdb(
    env: lmdb.Environment,
    expected_ids: list[str],
    dctx: zstd.ZstdDecompressor,
) -> bool:
    """Verify the integrity of all entries in the LMDB.

    Checks that every expected PDB ID is present and that every stored value
    can be decompressed without error.

    Args:
        env: An open LMDB environment.
        expected_ids: List of PDB IDs that should be present.
        dctx: A zstd decompressor instance.

    Returns:
        True if all entries are valid, False otherwise.
    """
    with env.begin() as txn:
        stat = env.stat()
        logger.info(f"Verifying {stat['entries']} entries...")

        missing: list[str] = []
        corrupt: list[str] = []

        for pdb_id in expected_ids:
            val = txn.get(pdb_id.encode("utf-8"))
            if val is None:
                missing.append(pdb_id)
                continue
            try:
                data = dctx.decompress(val)
                if len(data) == 0:
                    corrupt.append(pdb_id)
            except Exception:
                corrupt.append(pdb_id)

    if missing:
        logger.error(f"  Missing: {len(missing)} entries")
        for pid in missing[:10]:
            logger.error(f"    {pid}")
    if corrupt:
        logger.error(f"  Corrupt: {len(corrupt)} entries")
        for pid in corrupt[:10]:
            logger.error(f"    {pid}")

    ok = not missing and not corrupt
    if ok:
        logger.info("  All entries valid.")
    return ok


# ---
# Main
# ---


async def _run(
    binders_path: Path,
    output_path: Path,
    max_entries: int | None,
    concurrency: int,
) -> None:
    """Async entry point for the PDB structure download pipeline.

    Args:
        binders_path: Path to the binders CSV file.
        output_path: LMDB directory path for storing structures.
        max_entries: Optional cap on the number of entries to download.
        concurrency: Number of concurrent HTTP downloads.
    """
    pdb_ids = collect_pdb_ids(binders_path)
    if max_entries is not None:
        pdb_ids = pdb_ids[:max_entries]

    output_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_path), map_size=LMDB_MAP_SIZE)

    existing = get_existing_ids(env)
    to_download = [pid for pid in pdb_ids if pid not in existing]

    logger.info(
        f"{len(pdb_ids)} PDB IDs, {len(existing)} cached, "
        f"{len(to_download)} to download"
    )

    if to_download:
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        semaphore = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as client:
            tasks = [
                download_one(client, semaphore, pid)
                for pid in to_download
            ]

            done = 0
            ok = 0
            failed: list[tuple[str, str]] = []
            log_interval = max(1, len(tasks) // 20)

            for coro in asyncio.as_completed(tasks):
                pdb_id, content, msg = await coro
                done += 1

                if content is not None:
                    compressed = cctx.compress(content)
                    with env.begin(write=True) as txn:
                        txn.put(pdb_id.encode("utf-8"), compressed)
                    ok += 1
                else:
                    failed.append((pdb_id, msg))

                if done % log_interval == 0 or done == len(tasks):
                    logger.info(f"  [{done}/{len(tasks)}] ok={ok} fail={len(failed)}")

        logger.info(f"Download complete: {ok} new, {len(failed)} failed")
        for pid, reason in failed[:20]:
            logger.error(f"  FAIL {pid}: {reason}")

    dctx = zstd.ZstdDecompressor()
    verify_lmdb(env, pdb_ids, dctx)

    env.close()


def download_pdb_structures(
    binders_path: Path,
    output_path: Path,
    max_entries: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    """Download PDB structures from RCSB into LMDB + zstd store.

    Args:
        binders_path: Path to the binders CSV file.
        output_path: LMDB directory path for storing structures.
        max_entries: Optional cap on the number of entries to download.
        concurrency: Number of concurrent HTTP downloads.
    """
    asyncio.run(_run(binders_path, output_path, max_entries, concurrency))


def main() -> None:
    """Parse CLI arguments and initiate PDB structure downloads."""
    parser = argparse.ArgumentParser(
        description="Download PDB structures from RCSB into LMDB + zstd store"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Max entries to download"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent downloads (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config)

    if not config.binders_pdb.exists():
        logger.error(f"Input file not found: {config.binders_pdb}")
        sys.exit(1)

    download_pdb_structures(
        binders_path=config.binders_pdb,
        output_path=config.structures_pdb,
        max_entries=args.max,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()

"""
Inspect LMDB Databases

A utility to explore MÍMIR's LMDB databases. Supports both the structure
cache (zstd-compressed mmCIF) and the feature database (msgpack-encoded dicts).

Usage:
    uv run python -m scripts.dataset.inspect_lmdb data/run78-v2/structures_pdb --list
    uv run python -m scripts.dataset.inspect_lmdb data/run78-v2/features_binders --key 1OBY_P
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import zstandard as zstd


logger = logging.getLogger(__name__)


# ---
# Core Logic
# ---

def inspect_db(
    db_path: Path,
    key: str | None = None,
    list_keys: bool = False,
    head: int | None = None,
    full: bool = False,
) -> None:
    """Inspect an LMDB database.

    Args:
        db_path: Path to the LMDB directory.
        key: Specific key to inspect.
        list_keys: Whether to list all keys.
        head: Limit for listing keys.
        full: Whether to print full values without truncation.
    """
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    
    try:
        with env.begin() as txn:
            stat = env.stat()
            print(f"--- Database Stats: {db_path.name} ---")
            print(f"Entries: {stat['entries']}")
            print(f"Btree Depth: {stat['depth']}")
            print(f"Branch Pages: {stat['branch_pages']}")
            print(f"Leaf Pages: {stat['leaf_pages']}")
            print("-" * (len(db_path.name) + 24))

            if list_keys:
                print("\nKeys:")
                cursor = txn.cursor()
                count = 0
                for k in cursor.iternext(values=False):
                    print(f"  {k.decode('utf-8', errors='replace')}")
                    count += 1
                    if head and count >= head:
                        print(f"  ... (limited to {head} keys)")
                        break
            
            if key:
                val = txn.get(key.encode("utf-8"))
                
                if val is None:
                    print(f"\nKey '{key}' not found.")
                else:
                    print(f"\nEntry: {key}")
                    print(f"Raw Size: {len(val)} bytes")
                    
                    decoded = False

                    if val.startswith(b"\x28\xb5\x2f\xfd"):
                        print("Encoding: Zstandard (likely structures)")
                        try:
                            dctx = zstd.ZstdDecompressor()
                            decompressed = dctx.decompress(val)
                            print(f"Decompressed Size: {len(decompressed)} bytes")
                            
                            if full:
                                print(f"Content:\n{decompressed.decode('utf-8', errors='replace')}")
                            else:
                                preview = decompressed[:200].decode("utf-8", errors="replace")
                                print(f"Content Preview:\n{preview}...")
                            decoded = True
                        except Exception as e:
                            print(f"Failed to decompress Zstd: {e}")

                    if not decoded:
                        try:
                            data = msgpack.unpackb(val)
                            print("Encoding: Msgpack (likely features)")
                            if isinstance(data, dict):
                                for k, v in data.items():
                                    if not full:
                                        if isinstance(v, list) and len(v) > 5:
                                            print(f"  {k}: list[len={len(v)}] {v[:3]}...{v[-2:]}")
                                        elif isinstance(v, (str, bytes)) and len(v) > 100:
                                            print(f"  {k}: {type(v).__name__}[len={len(v)}] {v[:50]}...")
                                        else:
                                            print(f"  {k}: {v}")
                                    else:
                                        print(f"  {k}: {v}")
                            else:
                                print(f"  Data: {data}")
                            decoded = True
                        except Exception:
                            pass
                    
                    if not decoded:
                        print("Encoding: Unknown (Raw Bytes)")
                        print(f"Hex Preview: {val[:32].hex()}...")

    finally:
        env.close()


# ---
# Main
# ---

def main() -> None:
    """Parse CLI arguments and run the LMDB inspector utility."""
    parser = argparse.ArgumentParser(description="Inspect MÍMIR LMDB databases")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the LMDB directory",
    )
    parser.add_argument(
        "--key",
        type=str,
        help="Specific key to inspect and decode",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all keys in the database",
    )
    parser.add_argument(
        "--head",
        type=int,
        help="Limit number of keys listed",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full values without truncation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Log progress and info",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.path.exists():
        logger.error(f"Database path does not exist: {args.path}")
        sys.exit(1)

    inspect_db(args.path, key=args.key, list_keys=args.list, head=args.head, full=args.full)


if __name__ == "__main__":
    main()

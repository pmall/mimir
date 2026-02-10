"""
Generate Viral Binders Dataset

Extracts viral binders from virus-human (VH) interactions.
Small proteins with no mapping are considered as binding sequences.

Usage:
    uv run python -m scripts.dataset.generate_viral_binders [--verbose] [--min-length 4] [--max-length 512]
"""

import argparse
import csv
import sys
from pathlib import Path

from .utils import (
    get_db_connection,
    extract_binders_from_mapping,
    extract_binder_from_empty_mapping,
)


def generate_viral_binders(min_len: int = 4, max_len: int = 512, verbose: bool = False) -> None:
    """Extract viral binders from VH interactions and write to CSV.

    This function fetches virus-human interactions from the database, filters them
    by length and mapping validity, and writes the resulting binder sequences to a CSV file.

    Args:
        min_len: Minimum sequence length (inclusive).
        max_len: Maximum sequence length (inclusive).
        verbose: Whether to print progress and statistics to stderr.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # 1. Fetch Data
        if verbose:
            print(f"Fetching VH interactions ({min_len}-{max_len} aa)...", file=sys.stderr)

        cursor.execute(
            """
            SELECT 
                d.accession1,
                d.accession2, d.start2, d.stop2,
                d.mapping2,
                p2.sequences as sequences2
            FROM dataset d
            JOIN proteins p2 ON p2.id = d.protein2_id
            WHERE d.type = 'vh'
              AND d.is_obsolete1 = false
              AND d.is_obsolete2 = false
              AND d.deleted_at IS NULL
            """
        )

        # associations = set of (target, binder_data) tuples
        associations = set()
        rows_processed = 0

        # 2. Process Binders
        for row in cursor:
            (
                accession1,
                accession2, start2, stop2,
                mapping2,
                sequences2,
            ) = row

            rows_processed += 1

            # mapping2: binders from viral protein2, target is human protein1
            if mapping2:
                for binder_data in extract_binders_from_mapping(
                    mapping2, accession2, start2, stop2, sequences2, min_len, max_len
                ):
                    source_acc, prot_start, prot_stop, occ_start, occ_stop, seq = binder_data
                    associations.add((accession1, source_acc, prot_start, prot_stop, occ_start, occ_stop, seq))
            else:
                binder_data = extract_binder_from_empty_mapping(
                    accession2, start2, stop2, sequences2, min_len, max_len
                )
                if binder_data:
                    source_acc, prot_start, prot_stop, occ_start, occ_stop, seq = binder_data
                    associations.add((accession1, source_acc, prot_start, prot_stop, occ_start, occ_stop, seq))

        # 3. Write Output
        # Calculate stats
        # a = (target, source_acc, prot_start, prot_stop, occ_start, occ_stop, seq)
        unique_targets = {a[0] for a in associations}
        unique_sources = {(a[1], a[2], a[3]) for a in associations}
        unique_binders = {(a[1], a[2], a[3], a[4], a[5], a[6]) for a in associations}
        unique_sequences = {a[6] for a in associations}

        if verbose:
            print(f"Processed {rows_processed} VH rows", file=sys.stderr)
            print(f"  Associations: {len(associations)}", file=sys.stderr)
            print(f"  Unique binders: {len(unique_binders)}", file=sys.stderr)
            print(f"  Unique sequences: {len(unique_sequences)}", file=sys.stderr)
            print(f"  Unique sources: {len(unique_sources)}", file=sys.stderr)
            print(f"  Unique human targets: {len(unique_targets)}", file=sys.stderr)

        # Write output
        output_dir = Path(__file__).parent.parent.parent / "data" / "run78-v2"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "viral_binders.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "target",
                "source_accession",
                "protein_start",
                "protein_stop",
                "occ_start",
                "occ_stop",
                "sequence",
            ])

            for association in sorted(associations, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5])):
                writer.writerow(association)

        if verbose:
            print(f"Written to {output_path}", file=sys.stderr)

        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Viral Binders Dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output statistics")
    parser.add_argument("--min-length", type=int, default=4, help="Minimum sequence length (default: 4)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: 512)")
    args = parser.parse_args()

    generate_viral_binders(min_len=args.min_length, max_len=args.max_length, verbose=args.verbose)

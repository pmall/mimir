"""
Generate Viral Binders Dataset

Extracts viral binders from virus-human (VH) interactions.
Small proteins with no mapping are considered as binding sequences.

Usage:
    uv run python -m scripts.dataset.generate_viral_binders [--verbose] [--min-length 4] [--max-length 512] [--output PATH]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from scripts.dataset.utils import (
    extract_binder_from_empty_mapping,
    extract_binders_from_mapping,
    generate_structure_id,
    get_db_connection,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_viral_binders(min_len: int = 4, max_len: int = 512, verbose: bool = False, output: Path | None = None) -> None:
    """Extract viral binders from VH interactions and write to CSV.

    This function fetches virus-human interactions from the database, filters them
    by length and mapping validity, and writes the resulting binder sequences to a CSV file.

    Args:
        min_len: Minimum sequence length (inclusive).
        max_len: Maximum sequence length (inclusive).
        verbose: Whether to print progress and statistics to stderr.
        output: Output CSV path. Defaults to data/run78-v2/viral_binders.csv.
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

        # 3. Process into unified format
        # Group by (target, sequence)
        grouped: dict[tuple[str, str], list[tuple]] = {}
        seen_ids = set()

        for assoc in associations:
            # assoc: (target, source_acc, prot_start, prot_stop, occ_start, occ_stop, sequence)
            target = assoc[0]
            sequence = assoc[6]
            key = (target, sequence)
            grouped.setdefault(key, []).append(assoc)

        final_rows: list[dict] = []
        for (target, sequence), group in grouped.items():
            # Generate deterministic structure ID from target + sequence
            struct_id = generate_structure_id("V:", target, sequence)
            
            if struct_id in seen_ids:
                raise ValueError(f"Collision detected! Structure ID {struct_id} already exists (Target: {target}, Seq: {sequence})")
            seen_ids.add(struct_id)
            
            # Build sources list
            sources = []
            for item in group:
                # item: (target, source_acc, prot_start, prot_stop, occ_start, occ_stop, sequence)
                sources.append({
                    "uniprot": item[1],
                    "protein_start": item[2],
                    "protein_stop": item[3],
                    "occ_start": item[4],
                    "occ_stop": item[5],
                })
                
            final_rows.append({
                "type": "VH",
                "target": target,
                "sequence": sequence,
                "structure_id": struct_id,
                "sources": json.dumps(sources),
            })

        if verbose:
            print(f"Processed {rows_processed} VH rows", file=sys.stderr)
            print(f"  Raw associations: {len(associations)}", file=sys.stderr)
            print(f"  Unique (target, sequence) pairs: {len(final_rows)}", file=sys.stderr)

        if output is None:
            output = Path(__file__).parent.parent.parent / "data" / "run78-v2" / "viral_binders.csv"
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["type", "target", "sequence", "structure_id", "sources"])
            writer.writeheader()
            
            final_rows.sort(key=lambda x: (x["target"], x["sequence"]))
            writer.writerows(final_rows)

        if verbose:
            print(f"Written to {output}", file=sys.stderr)

        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Viral Binders Dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output statistics")
    parser.add_argument("--min-length", type=int, default=4, help="Minimum sequence length (default: 4)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: 512)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path (default: data/run78-v2/viral_binders.csv)")
    args = parser.parse_args()

    generate_viral_binders(min_len=args.min_length, max_len=args.max_length, verbose=args.verbose, output=args.output)

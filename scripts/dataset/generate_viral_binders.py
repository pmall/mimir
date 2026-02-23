"""
Generate Viral Binders Dataset

Extracts viral binders from virus-human (VH) interactions.
Small proteins with no mapping are considered as binding sequences.

Usage:
    uv run python -m scripts.dataset.generate_viral_binders \\
        -o data/run78-v2/binders_lists/viral_binders_96aa.csv \\
        [--min-length 4] [--max-length 96] [--verbose]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from scripts.dataset.utils import (
    extract_binder_from_empty_mapping,
    extract_binders_from_mapping,
    generate_structure_id,
    get_db_connection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_viral_binders(
    output: Path,
    min_len: int = 4,
    max_len: int = 512,
    verbose: bool = False,
) -> None:
    """Extract viral binders from VH interactions and write to CSV.

    Fetches virus-human interactions from the database, filters by length
    and mapping validity, and writes the resulting binder sequences to a CSV.

    Args:
        output: Output CSV path.
        min_len: Minimum binder sequence length (inclusive).
        max_len: Maximum binder sequence length (inclusive).
        verbose: Whether to log progress and statistics.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # 1. Fetch Data
        logger.info(f"Fetching VH interactions ({min_len}-{max_len} aa)...")

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

        # 3. Group by (target, sequence)
        grouped: dict[tuple[str, str], list[tuple]] = {}
        seen_ids: set[str] = set()

        for assoc in associations:
            # assoc: (target, source_acc, prot_start, prot_stop, occ_start, occ_stop, sequence)
            key = (assoc[0], assoc[6])
            grouped.setdefault(key, []).append(assoc)

        final_rows: list[dict] = []
        for (target, sequence), group in grouped.items():
            struct_id = generate_structure_id("V:", target, sequence)

            if struct_id in seen_ids:
                raise ValueError(
                    f"Collision detected: {struct_id} (target={target}, seq={sequence})"
                )
            seen_ids.add(struct_id)

            sources = [
                {
                    "uniprot": item[1],
                    "protein_start": item[2],
                    "protein_stop": item[3],
                    "occ_start": item[4],
                    "occ_stop": item[5],
                }
                for item in group
            ]

            final_rows.append({
                "type": "VH",
                "target": target,
                "sequence": sequence,
                "structure_id": "",
                "binder_id": struct_id,
                "sources": json.dumps(sources),
            })

        logger.info(f"Processed {rows_processed} VH rows")
        logger.info(f"  Raw associations:             {len(associations)}")
        logger.info(f"  Unique (target, sequence) pairs: {len(final_rows)}")

        # 4. Write output
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["type", "target", "sequence", "structure_id", "binder_id", "sources"]
            )
            writer.writeheader()
            final_rows.sort(key=lambda x: (x["target"], x["sequence"]))
            writer.writerows(final_rows)

        logger.info(f"Written to {output}")

        cursor.close()
    finally:
        conn.close()


def main() -> None:
    """Parse CLI arguments and generate the viral binders CSV dataset."""
    parser = argparse.ArgumentParser(description="Generate Viral Binders Dataset")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="Minimum binder sequence length (default: 4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum binder sequence length (default: 512)",
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

    generate_viral_binders(
        output=args.output,
        min_len=args.min_length,
        max_len=args.max_length,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

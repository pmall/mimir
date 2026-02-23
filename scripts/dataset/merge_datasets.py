"""
Merge multiple binder datasets with priority-based deduplication.

Takes multiple CSV files sharing the unified schema (type, target, sequence,
structure_id, sources). Deduplicates on (target, sequence) pairs, with earlier
files taking priority over later ones.

Usage:
    uv run python -m scripts.dataset.merge_datasets \\
        data/run78-v2/pdb_binders.csv data/run78-v2/human_binders.csv \\
        -o data/run78-v2/merged.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

FIELDNAMES = ["type", "target", "sequence", "structure_id", "binder_id", "sources"]


# ---
# Main
# ---


def merge_datasets(
    input_files: list[Path],
    output: Path,
) -> None:
    """Merge multiple binder CSVs with first-file-priority deduplication.

    Iterates input files in order. The first file to introduce a (target,
    sequence) pair defines the row (type, structure_id). Sources from all
    files are merged into a single JSON array.

    Args:
        input_files: Ordered list of input CSV paths.
        output: Output CSV path.
        verbose: Whether to log statistics.
    """
    merged: dict[tuple[str, str], dict] = {}

    for path in input_files:
        added = 0
        extended = 0

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["target"], row["sequence"])
                sources = json.loads(row["sources"])
                if key not in merged:
                    row["sources"] = sources
                    merged[key] = row
                    added += 1
                else:
                    merged[key]["sources"].extend(sources)
                    extended += 1

        logger.info(f"{path}: {added} new, {extended} sources merged")

    rows = list(merged.values())
    for row in rows:
        row["sources"].sort(key=lambda x: json.dumps(x, sort_keys=True))
        row["sources"] = json.dumps(row["sources"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Total: {len(rows)} unique (target, sequence) pairs written to {output}")


def main() -> None:
    """Parse CLI arguments and run the dataset merge utility."""
    parser = argparse.ArgumentParser(
        description="Merge binder datasets with priority-based deduplication"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input CSV files (earlier files take priority)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path",
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

    for path in args.inputs:
        if not path.exists():
            logger.error(f"Input file not found: {path}")
            sys.exit(1)

    merge_datasets(input_files=args.inputs, output=args.output)


if __name__ == "__main__":
    main()

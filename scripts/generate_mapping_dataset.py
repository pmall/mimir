"""
Generate Mapping Dataset (4-512 aa)

Generates 'data/mapping_dataset.csv' containing interacting sequences with length between 4 and 512 amino acids.
Usage:
    uv run python scripts/generate_mapping_dataset.py [--verbose]
"""

import argparse
import csv
import sys
from pathlib import Path
from dataset_utils import get_db_connection, get_all_interacting_pairs

MIN_LEN = 4
MAX_LEN = 512

def generate_mapping_dataset(verbose=False):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        if verbose:
            print(f"Generating mapping dataset ({MIN_LEN}-{MAX_LEN} aa)...", file=sys.stderr)

        interacting_pairs, _ = get_all_interacting_pairs(cursor, MIN_LEN, MAX_LEN, verbose)

        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "mapping_dataset.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target", "sequence"])

            for target, sequence in sorted(interacting_pairs):
                writer.writerow([target, sequence])

        if verbose:
            print(f"\nMapping Dataset written to {output_path}", file=sys.stderr)
            print(f"Total pairs: {len(interacting_pairs)}", file=sys.stderr)

        cursor.close()
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mapping Dataset (4-512aa)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output statistics")
    args = parser.parse_args()

    generate_mapping_dataset(verbose=args.verbose)

"""
Dataset Generation Script for Protein-Protein Interaction Data

This script extracts interacting peptide sequences from a PostgreSQL database.
It is the first step in the data pipeline, creating a CSV file used for training.

Overview:
    1. Connects to the PostgreSQL database using environment variables.
    2. Queries for "Human-Human" (hh) and "Virus-Human" (vh) interactions.
    3. Extracts peptide sequences from mapping JSONs.
    4. Filters peptides by length (4-20 amino acids).
    5. Writes unique (target, peptide) pairs to 'data/dataset.csv'.

Output Format (CSV):
    - target: Accession ID of the target protein.
    - sequence: The interacting peptide sequence.

Usage:
    uv run python scripts/generate_dataset.py [--verbose] [--seed 42]
"""

import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

MIN_PEPTIDE_LENGTH = 4
MAX_PEPTIDE_LENGTH = 20


def get_db_connection():
    """Create a PostgreSQL database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def extract_peptides_from_mapping(mapping):
    """
    Extract valid peptide sequences from a mapping JSON array.

    Yields peptides with length between MIN_PEPTIDE_LENGTH and MAX_PEPTIDE_LENGTH.
    """
    if not mapping:
        return

    for item in mapping:
        sequence = item.get("sequence", "")
        if MIN_PEPTIDE_LENGTH <= len(sequence) <= MAX_PEPTIDE_LENGTH:
            yield sequence


def get_hh_interacting_pairs(cursor):
    """
    Fetch human-human interacting peptide pairs.

    For hh interactions:
    - mapping1 peptides (from protein1) target accession2
    - mapping2 peptides (from protein2) target accession1
    """
    cursor.execute(
        """
        SELECT accession1, accession2, mapping1, mapping2
        FROM dataset
        WHERE type = 'hh'
          AND is_obsolete1 = false
          AND is_obsolete2 = false
          AND deleted_at IS NULL
    """
    )

    pairs = set()
    for accession1, accession2, mapping1, mapping2 in cursor:
        for peptide in extract_peptides_from_mapping(mapping1):
            pairs.add((accession2, peptide))
        for peptide in extract_peptides_from_mapping(mapping2):
            pairs.add((accession1, peptide))

    return pairs


def get_vh_interacting_pairs(cursor):
    """
    Fetch virus-human interacting peptide pairs.

    For vh interactions, mapping2 peptides (from viral protein) target accession1 (human).
    """
    cursor.execute(
        """
        SELECT accession1, mapping2
        FROM dataset
        WHERE type = 'vh'
          AND is_obsolete1 = false
          AND is_obsolete2 = false
          AND deleted_at IS NULL
          AND mapping2 IS NOT NULL
          AND mapping2::text != '[]'
    """
    )

    pairs = set()
    for human_accession, mapping2 in cursor:
        for peptide in extract_peptides_from_mapping(mapping2):
            pairs.add((human_accession, peptide))

    return pairs


def get_all_interacting_pairs(cursor, verbose=False):
    """
    Fetch all interacting peptide pairs from both vh and hh interactions.

    Returns:
        tuple: (unique_pairs, unique_peptides)
    """
    hh_pairs = get_hh_interacting_pairs(cursor)
    vh_pairs = get_vh_interacting_pairs(cursor)

    unique_pairs = hh_pairs | vh_pairs
    unique_peptides = {peptide for _, peptide in unique_pairs}

    if verbose:
        hh_peptides = {peptide for _, peptide in hh_pairs}
        vh_peptides = {peptide for _, peptide in vh_pairs}
        print(
            f"  Human peptides: {len(hh_peptides)} unique, {len(hh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Viral peptides: {len(vh_peptides)} unique, {len(vh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Total: {len(unique_peptides)} unique, {len(unique_pairs)} pairs",
            file=sys.stderr,
        )

    return unique_pairs, unique_peptides


def generate_dataset(verbose=False):
    """Generate the complete dataset and write to data/dataset.csv."""
    import csv
    from pathlib import Path

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        if verbose:
            print("Fetching interacting pairs...", file=sys.stderr)

        interacting_pairs, _ = get_all_interacting_pairs(
            cursor, verbose
        )

        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "dataset.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target", "sequence"])

            for target, peptide in sorted(interacting_pairs):
                writer.writerow([target, peptide])

        if verbose:
            print(f"\nDataset written to {output_path}", file=sys.stderr)
            print("\n--- Statistics ---", file=sys.stderr)
            print(f"Interacting pairs: {len(interacting_pairs)}", file=sys.stderr)

        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PPI dataset from PostgreSQL")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Output statistics"
    )
    args = parser.parse_args()

    generate_dataset(verbose=args.verbose)

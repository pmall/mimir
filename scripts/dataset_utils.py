"""
Shared Dataset Utilities

Common functions for extracting and filtering peptide sequences from the database.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Create a PostgreSQL database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def extract_peptides_from_mapping(mapping, min_len, max_len):
    """
    Extract valid peptide sequences from a mapping JSON array with length filtering.
    """
    if not mapping:
        return

    for item in mapping:
        sequence = item.get("sequence", "")
        if min_len <= len(sequence) <= max_len:
            yield sequence


def get_hh_interacting_pairs(cursor, min_len, max_len):
    """
    Fetch human-human interacting peptide pairs.
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
        for peptide in extract_peptides_from_mapping(mapping1, min_len, max_len):
            pairs.add((accession2, peptide))
        for peptide in extract_peptides_from_mapping(mapping2, min_len, max_len):
            pairs.add((accession1, peptide))

    return pairs


def get_vh_interacting_pairs(cursor, min_len, max_len):
    """
    Fetch virus-human interacting peptide pairs.
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
        for peptide in extract_peptides_from_mapping(mapping2, min_len, max_len):
            pairs.add((human_accession, peptide))

    return pairs


def get_all_interacting_pairs(cursor, min_len, max_len, verbose=False):
    """
    Fetch all interacting peptide pairs from both vh and hh interactions
    within the specified length range.

    Returns:
        tuple: (unique_pairs, unique_peptides)
    """
    hh_pairs = get_hh_interacting_pairs(cursor, min_len, max_len)
    vh_pairs = get_vh_interacting_pairs(cursor, min_len, max_len)

    unique_pairs = hh_pairs | vh_pairs
    unique_peptides = {peptide for _, peptide in unique_pairs}

    if verbose:
        hh_peptides = {peptide for _, peptide in hh_pairs}
        vh_peptides = {peptide for _, peptide in vh_pairs}
        print(
            f"  Human peptides ({min_len}-{max_len}): {len(hh_peptides)} unique, {len(hh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Viral peptides ({min_len}-{max_len}): {len(vh_peptides)} unique, {len(vh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Total ({min_len}-{max_len}): {len(unique_peptides)} unique, {len(unique_pairs)} pairs",
            file=sys.stderr,
        )

    return unique_pairs, unique_peptides

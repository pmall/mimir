"""
Shared utilities for v2 dataset generation.
"""

import hashlib
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection() -> psycopg2.extensions.connection:
    """Create a PostgreSQL database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def get_canonical_sequence(
    sequences: dict[str, str], accession: str, start: int, stop: int
) -> str:
    """
    Get the canonical sequence sliced by start:stop coordinates.

    Args:
        sequences: Dict of accession -> sequence from proteins.sequences
        accession: The canonical accession
        start: Start coordinate (1-indexed)
        stop: Stop coordinate (1-indexed, inclusive)

    Returns:
        The sliced canonical sequence
    """
    full_seq = sequences.get(accession, "")
    return full_seq[start - 1 : stop]


def extract_binders_from_mapping(
    mapping: list,
    canonical_accession: str,
    canonical_start: int,
    canonical_stop: int,
    sequences: dict[str, str],
    min_len: int = 4,
    max_len: int = 512,
) -> list[tuple]:
    """
    Extract binders from a mapping JSON array.

    Only considers canonical sequences, ignores isoforms.

    Filters:
    - Sequence length must be between min_len and max_len
    - Only single occurrence on canonical (skip multi-occurrence)
    - Ignores isoforms entirely

    Args:
        mapping: The mapping JSON array
        canonical_accession: The canonical protein accession
        canonical_start: Start coordinate of canonical
        canonical_stop: Stop coordinate of canonical
        sequences: Dict of accession -> sequence from proteins.sequences
        min_len: Minimum sequence length (default 4)
        max_len: Maximum sequence length (default 512)

    Returns:
        List of tuples: (source_accession, protein_start, protein_stop, occ_start, occ_stop, sequence)
    """
    if not mapping:
        return []

    binders = []
    for item in mapping:  # type: dict
        sequence = item.get("sequence", "")
        if not (min_len <= len(sequence) <= max_len):
            continue

        isoforms = item.get("isoforms", [])
        for isoform in isoforms:
            isoform_acc = isoform.get("accession", "")

            # Only consider canonical, skip isoforms
            if isoform_acc != canonical_accession:
                continue

            occurrences = isoform.get("occurrences", [])

            # Skip if multiple occurrences
            if len(occurrences) != 1:
                continue

            occ = occurrences[0]
            occ_start = int(occ.get("start"))
            occ_stop = int(occ.get("stop"))

            binders.append(
                (
                    canonical_accession,
                    canonical_start,
                    canonical_stop,
                    occ_start,
                    occ_stop,
                    sequence,
                )
            )

    return binders


def extract_binder_from_empty_mapping(
    canonical_accession: str,
    canonical_start: int,
    canonical_stop: int,
    sequences: dict[str, str],
    min_len: int = 4,
    max_len: int = 512,
) -> tuple | None:
    """
    Extract binder from canonical sequence when mapping is empty.

    Only canonical sequences are considered (not isoforms).

    Args:
        canonical_accession: The canonical protein accession
        canonical_start: Start coordinate of canonical
        canonical_stop: Stop coordinate of canonical
        sequences: Dict of accession -> sequence from proteins.sequences
        min_len: Minimum sequence length (default 4)
        max_len: Maximum sequence length (default 512)

    Returns:
        Tuple (source_accession, protein_start, protein_stop, occ_start, occ_stop, sequence)
        or None if length constraints not met
    """
    sequence = get_canonical_sequence(
        sequences, canonical_accession, canonical_start, canonical_stop
    )

    if not (min_len <= len(sequence) <= max_len):
        return None

    return (
        canonical_accession,
        canonical_start,
        canonical_stop,
        1,
        len(sequence),
        sequence,
    )


def generate_structure_id(prefix: str, target: str, sequence: str) -> str:
    """Generate a deterministic ID (e.g. H:A1B2C3D4) from the target and sequence.

    Args:
        prefix: Prefix for the ID (e.g. "H:" or "V:")
        target: The target accession
        sequence: The peptide sequence (will be upper-cased and stripped)

    Returns:
        Deterministic ID string.
    """
    normalized_data = f"{target}_{sequence.strip()}".upper()
    seq_hash = hashlib.sha256(normalized_data.encode("utf-8")).hexdigest()
    return f"{prefix}{seq_hash[:12].upper()}"




"""
Shared Dataset Utilities

Common functions for extracting and filtering binding sequences from the database.
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


def extract_sequences_from_mapping(mapping, min_len, max_len):
    """
    Extract valid binding sequences from a mapping JSON array with length filtering.
    """
    if not mapping:
        return

    for item in mapping:
        sequence = item.get("sequence", "")
        if min_len <= len(sequence) <= max_len:
            yield sequence


def get_hh_interacting_pairs(cursor, min_len, max_len):
    """
    Fetch human-human interacting sequence pairs.
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
        for sequence in extract_sequences_from_mapping(mapping1, min_len, max_len):
            pairs.add((accession2, sequence))
        for sequence in extract_sequences_from_mapping(mapping2, min_len, max_len):
            pairs.add((accession1, sequence))

    return pairs


def get_vh_interacting_pairs(cursor, min_len, max_len):
    """
    Fetch virus-human interacting sequence pairs.
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
        for sequence in extract_sequences_from_mapping(mapping2, min_len, max_len):
            pairs.add((human_accession, sequence))

    return pairs


def get_all_interacting_pairs(cursor, min_len, max_len, verbose=False):
    """
    Fetch all interacting sequence pairs from both vh and hh interactions
    within the specified length range.

    Returns:
        tuple: (unique_pairs, unique_sequences)
    """
    hh_pairs = get_hh_interacting_pairs(cursor, min_len, max_len)
    vh_pairs = get_vh_interacting_pairs(cursor, min_len, max_len)

    unique_pairs = hh_pairs | vh_pairs
    unique_sequences = {sequence for _, sequence in unique_pairs}

    if verbose:
        hh_sequences = {sequence for _, sequence in hh_pairs}
        vh_sequences = {sequence for _, sequence in vh_pairs}
        print(
            f"  Human binding sequences ({min_len}-{max_len}): {len(hh_sequences)} unique, {len(hh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Viral binding sequences ({min_len}-{max_len}): {len(vh_sequences)} unique, {len(vh_pairs)} pairs",
            file=sys.stderr,
        )
        print(
            f"  Total ({min_len}-{max_len}): {len(unique_sequences)} unique, {len(unique_pairs)} pairs",
            file=sys.stderr,
        )

    return unique_pairs, unique_sequences


def get_targets_with_domain(cursor, domain_keyword, verbose=False):
    """
    Fetch all targets (accession -> name) that have a domain description 
    containing the keyword (case insensitive).
    """
    if verbose:
        print(f"Fetching targets with {domain_keyword} domains...", file=sys.stderr)
    
    query = """
    SELECT accession, names 
    FROM proteins_versions, jsonb_array_elements(features) as value 
    WHERE value->>'type' = 'domain' 
    AND value->>'description' ILIKE %s
    """
    cursor.execute(query, (f'%{domain_keyword}%',))
    
    # Handle multiple names (take the first one)
    targets = {}
    for accession, names in cursor:
        name = names[0] if names else "N/A"
        targets[accession] = name
    
    if verbose:
        print(f"  Found {len(targets)} {domain_keyword} candidates in DB.", file=sys.stderr)
    return targets


def get_domain_targets_with_counts(domain_keyword, min_len, max_len, verbose=True):
    """
    Returns a list of targets containing the specified domain,
    including the count of associated binding sequences.
    
    Returns:
        list of tuples: (count, accession, name) sorted by count descending.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # 1. Get all interacting pairs
        if verbose:
            print(f"Fetching all interacting pairs ({min_len}-{max_len} aa)...", file=sys.stderr)
        interacting_pairs, _ = get_all_interacting_pairs(cursor, min_len, max_len, verbose=verbose)
        
        # Group sequences by target
        target_sequences = {}
        for target, sequence in interacting_pairs:
            if target not in target_sequences:
                target_sequences[target] = set()
            target_sequences[target].add(sequence)
            
        if verbose:
            print(f"Total unique targets involved in interactions: {len(target_sequences)}", file=sys.stderr)

        # 2. Fetch candidates from DB
        candidates = get_targets_with_domain(cursor, domain_keyword, verbose)
        
        # 3. Filter and Count
        results = []
        for accession, name in candidates.items():
            if accession in target_sequences:
                count = len(target_sequences[accession])
                results.append((count, accession, name))
        
        # 4. Sort (descending by count)
        results.sort(key=lambda x: x[0], reverse=True)
        
        return results

    finally:
        conn.close()

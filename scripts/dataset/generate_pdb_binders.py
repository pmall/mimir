"""
Generate PDB Binders Dataset

Queries the RCSB PDB Search API for binary protein-protein complexes
(exactly 2 protein entities, no DNA/RNA), involving at least one human protein.
Then batch-fetches entry + entity details via the GraphQL Data API to identify
human SwissProt targets and their binders (any species).

Groups by (target, sequence) pair, outputting a unified schema with JSON sources.

Usage:
    uv run python -m scripts.dataset.generate_pdb_binders [--verbose] [--min-length 4] [--max-length 512] [--output PATH]
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"

HUMAN_TAXONOMY_ID = 9606
PAGE_SIZE = 100
GRAPHQL_BATCH_SIZE = 50
MAX_CONCURRENT = 5

# Method ranking for deduplication (lower = better).
# When two entries have the same resolution, prefer X-ray over Cryo-EM over NMR.
METHOD_RANK: dict[str, int] = {
    "X-RAY DIFFRACTION": 0,
    "ELECTRON MICROSCOPY": 1,
    "SOLUTION NMR": 2,
    "SOLID-STATE NMR": 3,
    "NEUTRON DIFFRACTION": 4,
    "ELECTRON CRYSTALLOGRAPHY": 5,
}
DEFAULT_METHOD_RANK = 99
# Fallback resolution when unavailable (e.g. NMR). Set high so entries with
# actual resolution are preferred.
FALLBACK_RESOLUTION = 999.0


# ---------------------------------------------------------------------------
# Search phase
# ---------------------------------------------------------------------------


def build_search_query(start: int = 0) -> dict[str, Any]:
    """Build a Search API JSON query for binary protein-only entries.

    Filters:
    - Exactly 2 protein polymer entities
    - Exactly 2 total polymer entities (excludes DNA/RNA/hybrid)
    - At least one entity from Homo sapiens (taxonomy 9606)
    - Experimental structures only

    Args:
        start: Pagination offset.

    Returns:
        JSON-serialisable query dict.
    """
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count",
                        "operator": "equals",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                        "operator": "exact_match",
                        "value": "9606",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": start,
                "rows": PAGE_SIZE,
            },
            "results_content_type": ["experimental"],
        },
    }


async def fetch_all_entry_ids(client: httpx.AsyncClient, verbose: bool = False) -> list[str]:
    """Fetch all PDB entry IDs matching the binary protein complex filter.

    Args:
        client: An httpx async client.
        verbose: Whether to print progress.

    Returns:
        List of PDB entry IDs (e.g. ["1ABC", "2DEF", ...]).
    """
    entry_ids: list[str] = []
    start = 0
    total_count: int | None = None

    while True:
        query = build_search_query(start=start)
        response = await client.post(SEARCH_API_URL, json=query, timeout=60.0)
        response.raise_for_status()
        data = response.json()

        if total_count is None:
            total_count = data.get("total_count", 0)
            if verbose:
                print(f"Search API: {total_count} entries found", file=sys.stderr)

        result_set = data.get("result_set", [])
        if not result_set:
            break

        for result in result_set:
            entry_ids.append(result["identifier"])

        start += PAGE_SIZE
        if start >= total_count:
            break

    return entry_ids


# ---------------------------------------------------------------------------
# GraphQL enrichment phase
# ---------------------------------------------------------------------------

# Query entries with nested polymer entities. This gives us both:
# - Entry-level: experimental method + resolution (for best-entry selection)
# - Entity-level: sequence, taxonomy, UniProt accession, chain IDs
GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_entry_info {
      experimental_method
      resolution_combined
    }
    polymer_entities {
      rcsb_id
      entity_poly {
        pdbx_seq_one_letter_code_can
      }
      rcsb_entity_source_organism {
        ncbi_taxonomy_id
        ncbi_scientific_name
      }
      rcsb_polymer_entity_container_identifiers {
        asym_ids
        reference_sequence_identifiers {
          database_name
          database_accession
        }
      }
    }
  }
}
"""


def parse_entity(entity: dict[str, Any]) -> dict[str, Any]:
    """Parse a single polymer entity from a GraphQL response.

    Args:
        entity: One element from the polymer_entities array.

    Returns:
        Dict with: rcsb_id, taxonomy_id, organism_name,
        uniprot_accession, sequence, chain_ids.
    """
    info: dict = {
        "rcsb_id": entity.get("rcsb_id", ""),
        "taxonomy_id": None,
        "organism_name": "",
        "uniprot_accession": None,
        "sequence": "",
        "chain_ids": [],
    }

    source_organisms = entity.get("rcsb_entity_source_organism") or []
    if source_organisms:
        info["taxonomy_id"] = source_organisms[0].get("ncbi_taxonomy_id")
        info["organism_name"] = source_organisms[0].get("ncbi_scientific_name", "")

    container_ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    ref_seq_ids = container_ids.get("reference_sequence_identifiers") or []
    for ref in ref_seq_ids:
        if ref.get("database_name") == "UniProt":
            info["uniprot_accession"] = ref.get("database_accession")
            break

    info["chain_ids"] = container_ids.get("asym_ids") or []

    entity_poly = entity.get("entity_poly") or {}
    raw_seq = entity_poly.get("pdbx_seq_one_letter_code_can") or ""
    info["sequence"] = raw_seq.replace("\n", "").replace(" ", "")

    return info


def parse_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Parse a single entry from a GraphQL response.

    Args:
        entry: One element from the entries array.

    Returns:
        Dict with: entry_id, method, resolution, entities (list of parsed entity dicts).
    """
    entry_info = entry.get("rcsb_entry_info") or {}
    resolution_list = entry_info.get("resolution_combined") or []

    return {
        "entry_id": entry.get("rcsb_id", ""),
        "method": entry_info.get("experimental_method", ""),
        "resolution": resolution_list[0] if resolution_list else None,
        "entities": [
            parse_entity(e)
            for e in (entry.get("polymer_entities") or [])
            if e is not None
        ],
    }


async def fetch_entries_batch(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    entry_ids: list[str],
) -> list[dict]:
    """Fetch entry + entity details for a batch via GraphQL.

    Args:
        client: An httpx async client.
        semaphore: Concurrency limiter.
        entry_ids: List of PDB entry IDs.

    Returns:
        List of parsed entry dicts.
    """
    async with semaphore:
        response = await client.post(
            GRAPHQL_URL,
            json={"query": GRAPHQL_QUERY, "variables": {"ids": entry_ids}},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

    entries = data.get("data", {}).get("entries") or []
    return [parse_entry(e) for e in entries if e is not None]


# ---------------------------------------------------------------------------
# Processing + Deduplication
# ---------------------------------------------------------------------------


def extract_associations(
    entry: dict,
    min_len: int,
    max_len: int,
) -> list[dict]:
    """Extract binder associations from an entry.

    An association is valid when one entity is a human UniProt protein (target)
    and the other is a binder within the length bounds.

    Args:
        entry: Parsed entry dict from GraphQL.
        min_len: Minimum binder sequence length.
        max_len: Maximum binder sequence length.

    Returns:
        List of association dicts with all fields needed for output + dedup.
    """
    entities = entry["entities"]
    if len(entities) != 2:
        return []

    results: list[dict] = []

    for target_idx, binder_idx in [(0, 1), (1, 0)]:
        target = entities[target_idx]
        binder = entities[binder_idx]

        if target["taxonomy_id"] != HUMAN_TAXONOMY_ID:
            continue
        if not target["uniprot_accession"]:
            continue

        binder_seq = binder["sequence"]
        if not binder_seq:
            continue
        if not (min_len <= len(binder_seq) <= max_len):
            continue

        results.append({
            "target": target["uniprot_accession"],
            "pdb_id": entry["entry_id"],
            "binder_entity_id": binder["rcsb_id"].split("_")[-1],
            "binder_chain_ids": ",".join(binder["chain_ids"]),
            "binder_organism": binder["organism_name"],
            "binder_length": len(binder_seq),
            "sequence": binder_seq,
            "method": entry["method"],
            "resolution": entry["resolution"],
        })

    return results



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run(min_len: int, max_len: int, verbose: bool, output: Path | None) -> None:
    """Async entry point for the PDB binder retrieval pipeline.

    Args:
        min_len: Minimum binder sequence length.
        max_len: Maximum binder sequence length.
        verbose: Whether to print progress and statistics.
        output: Output CSV path. Defaults to data/run78-v2/pdb_binders.csv.
    """
    async with httpx.AsyncClient() as client:
        # 1. Search
        if verbose:
            print(f"Searching for binary protein complexes ({min_len}-{max_len} aa)...", file=sys.stderr)

        entry_ids = await fetch_all_entry_ids(client, verbose=verbose)

        if not entry_ids:
            if verbose:
                print("No entries found.", file=sys.stderr)
            return

        if verbose:
            print(f"Fetching {len(entry_ids)} entries via GraphQL...", file=sys.stderr)

        # 2. Batch fetch via GraphQL
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        batches = [
            entry_ids[i : i + GRAPHQL_BATCH_SIZE]
            for i in range(0, len(entry_ids), GRAPHQL_BATCH_SIZE)
        ]

        tasks = [
            fetch_entries_batch(client, semaphore, batch)
            for batch in batches
        ]

        all_entries: list[dict] = []
        completed = 0
        total_batches = len(batches)
        log_interval = max(1, total_batches // 10)
        for coro in asyncio.as_completed(tasks):
            entries = await coro
            all_entries.extend(entries)
            completed += 1
            if verbose and completed % log_interval == 0:
                pct = completed * 100 // total_batches
                print(f"  {pct}% ({len(all_entries)}/{len(entry_ids)})", file=sys.stderr)

        if verbose:
            print(f"  Done: {len(all_entries)} entries fetched", file=sys.stderr)

        # 3. Extract associations
        all_associations: list[dict] = []
        for entry in all_entries:
            all_associations.extend(extract_associations(entry, min_len, max_len))

        if verbose:
            print(f"  Raw associations: {len(all_associations)}", file=sys.stderr)

        # 4. Deduplicate by (target, sequence) to create entry + sources list
        # Key: (target, sequence)
        # Value: list of associations (all sources)
        grouped: dict[tuple[str, str], list[dict]] = {}
        for assoc in all_associations:
            key = (assoc["target"], assoc["sequence"])
            grouped.setdefault(key, []).append(assoc)

        # 5. Build final rows
        final_rows: list[dict] = []
        for (target, sequence), group in grouped.items():
            # Sort group by best resolution to pick the representative structure ID
            group.sort(
                key=lambda a: (
                    a["resolution"] if a["resolution"] is not None else FALLBACK_RESOLUTION,
                    METHOD_RANK.get(a["method"], DEFAULT_METHOD_RANK),
                )
            )
            best_entry = group[0]
            
            # Build sources list
            sources = []
            for item in group:
                sources.append({
                    "pdb_id": item["pdb_id"],
                    "chain": item["binder_chain_ids"],
                    "resolution": item["resolution"],
                    "method": item["method"],
                    "organism": item["binder_organism"],
                    "entity_id": item["binder_entity_id"],
                })

            final_rows.append({
                "type": "PDB",
                "target": target,
                "sequence": sequence,
                "structure_id": best_entry["pdb_id"],
                "sources": json.dumps(sources),
            })

        if verbose:
            print(f"After grouping:", file=sys.stderr)
            print(f"  Unique (target, sequence) pairs: {len(final_rows)}", file=sys.stderr)

        # 6. Write output
        if output is None:
            output = Path(__file__).parent.parent.parent / "data" / "run78-v2" / "pdb_binders.csv"
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["type", "target", "sequence", "structure_id", "sources"])
            writer.writeheader()
            
            final_rows.sort(key=lambda x: (x["target"], x["sequence"]))
            writer.writerows(final_rows)

        if verbose:
            print(f"Written to {output}", file=sys.stderr)


def generate_pdb_binders(
    min_len: int = 4,
    max_len: int = 512,
    verbose: bool = False,
    output: Path | None = None,
) -> None:
    """Query RCSB PDB for binary protein complexes and extract binder sequences.

    Finds PDB entries with exactly 2 protein entities where one is a human
    UniProt protein (the target) and the other is the binder (any species).
    Deduplicates by (target, sequence), storing all sources in a JSON array.

    Args:
        min_len: Minimum binder sequence length (inclusive).
        max_len: Maximum binder sequence length (inclusive).
        verbose: Whether to print progress and statistics to stderr.
        output: Output CSV path. Defaults to data/run78-v2/pdb_binders.csv.
    """
    asyncio.run(_run(min_len, max_len, verbose, output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDB Binders Dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output statistics")
    parser.add_argument("--min-length", type=int, default=4, help="Minimum sequence length (default: 4)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: 512)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path (default: data/run78-v2/pdb_binders.csv)")
    args = parser.parse_args()

    generate_pdb_binders(
        min_len=args.min_length,
        max_len=args.max_length,
        verbose=args.verbose,
        output=args.output,
    )

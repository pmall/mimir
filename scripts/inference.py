"""
MÍMIR v2 Inference Script.

Generates novel peptide binder sequences for a target protein using a
fine-tuned ESM-3 model with LoRA adapters. Uses Parallel Iterative Decoding:
the binder region starts fully masked, and the most-confident positions are
unmasked one batch at a time until the sequence is complete.

Usage:
    uv run python -m scripts.inference \\
        --input inference_params.tsv \\
        --output inference_results.tsv \\
        [-v]
"""

# ---------------------------------------------------------------------------
# Stdlib imports
# ---------------------------------------------------------------------------
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import lmdb
import msgpack
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from mimir.model import load_model
from mimir.tokenizer import MimirTokenizer, build_input_tensors, load_tokenizer

# ---------------------------------------------------------------------------
# Logger — configured in main(), silent at import time
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmdb").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_target_from_lmdb(
    lmdb_path: str,
    key: str,
    start_pos: int,
    end_pos: int,
) -> dict[str, Any]:
    """Reads and slices target structure features from an LMDB database.

    Args:
        lmdb_path: Path to the LMDB database file.
        key: UniProt accession used as the LMDB key.
        start_pos: 1-based start position of the target region (inclusive).
        end_pos: 1-based end position of the target region (inclusive).

    Returns:
        Dictionary with sliced fields: sequence, structure_tokens, sasa,
        coordinates.

    Raises:
        KeyError: If the key is not found in the database.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, map_size=LMDB_MAP_SIZE)
    with env.begin() as txn:
        raw = txn.get(key.encode("utf-8"))
    env.close()

    if raw is None:
        raise KeyError(f"Key '{key}' not found in LMDB at {lmdb_path}")

    record = msgpack.unpackb(raw, raw=False)

    # Convert 1-based inclusive [start_pos, end_pos] to 0-based Python slice
    s = start_pos - 1
    e = end_pos  # end_pos is inclusive, so slice [s:e] gives positions s..e-1 (0-based)

    return {
        "sequence": record["sequence"][s:e],
        "structure_tokens": record["structure_tokens"][s:e],
        "sasa": record["sasa"][s:e],
        "coordinates": record["coordinates"][s:e],
    }


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


def generate_sequences(
    model: torch.nn.Module,
    tokenizer: MimirTokenizer,
    input_tensors: tuple[torch.Tensor, ...],
    binder_len: int,
    num_sequences: int,
    unmask_per_iter: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    """Generates peptide sequences via Parallel Iterative Decoding.

    Starts with a fully masked binder region and iteratively unmasks the
    positions the model is most confident about (highest raw softmax
    probability), sampling the final token with temperature scaling.

    Args:
        model: ESM-3 model in eval mode.
        tokenizer: MimirTokenizer instance.
        input_tensors: Tuple of (seq, struct, sasa, sequence_id, chain_id,
            structure_coords) as returned by build_input_tensors.
        binder_len: Number of amino acids to generate.
        num_sequences: How many independent sequences to produce.
        unmask_per_iter: Tokens to unmask per decoding step.
        temperature: Sampling temperature (must be > 0).
        device: Torch device to run inference on.
        seed: Random seed for reproducibility.

    Returns:
        List of generated amino acid strings, each of length binder_len.

    Raises:
        ValueError: If temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    seq_template, struct_template, sasa_template, seq_id_template, chain_id_template, coords_template = input_tensors

    # Locate binder region boundaries in the template sequence tensor.
    # Layout: [BOS] [fp...] [CHAINBREAK] [binder...] [EOS]
    chainbreak_positions = (seq_template == tokenizer.seq_chainbreak).nonzero(as_tuple=True)[0]
    eos_positions = (seq_template == tokenizer.seq_eos).nonzero(as_tuple=True)[0]

    chainbreak_pos = chainbreak_positions[0].item()
    eos_pos = eos_positions[0].item()  # exclusive upper bound for slicing

    binder_start = chainbreak_pos + 1
    binder_end = eos_pos  # slice [binder_start:binder_end] gives exactly the binder tokens

    generated: list[str] = []

    # Match input dtypes to model parameters
    model_dtype = next(model.parameters()).dtype
    # Only use autocast if model is bfloat16 (usually on CUDA)
    use_autocast = (model_dtype == torch.bfloat16)

    with torch.inference_mode():
        for _ in range(num_sequences):
            # Fresh fully-masked working copy for each sequence
            seq = seq_template.clone()
            current_masked = binder_len

            while current_masked > 0:
                with torch.amp.autocast(device.type, dtype=model_dtype, enabled=use_autocast):
                    output = model(
                        sequence_tokens=seq.unsqueeze(0).to(device),
                        structure_tokens=struct_template.unsqueeze(0).to(device),
                        sasa_tokens=sasa_template.unsqueeze(0).to(device),
                        chain_id=chain_id_template.unsqueeze(0).to(device),
                        structure_coords=coords_template.unsqueeze(0).to(device, dtype=model_dtype),
                        sequence_id=seq_id_template.unsqueeze(0).to(device),
                    )

                # sequence_logits: [1, L, V]
                binder_logits = output.sequence_logits[0, binder_start:binder_end, :]  # [BL, V]

                # Raw confidence: softmax without temperature for position ranking
                raw_probs = F.softmax(binder_logits.float(), dim=-1)  # [BL, V]

                # Identify still-masked binder positions (relative to binder_start offset)
                binder_seq = seq[binder_start:binder_end]
                masked_positions = (binder_seq == tokenizer.seq_mask).nonzero(as_tuple=True)[0].tolist()

                if not masked_positions:
                    break

                # Rank masked positions by max probability (model confidence)
                max_probs = raw_probs[masked_positions].max(dim=-1).values  # [num_masked]
                top_k = min(unmask_per_iter, len(masked_positions))
                top_indices = torch.topk(max_probs, k=top_k).indices.tolist()
                positions_to_unmask = [masked_positions[i] for i in top_indices]

                for rel_pos in positions_to_unmask:
                    # Apply temperature at sampling time only
                    probs = F.softmax(binder_logits[rel_pos].float() / temperature, dim=-1)
                    token_id = torch.multinomial(probs, num_samples=1).item()
                    seq[binder_start + rel_pos] = token_id

                current_masked -= top_k

            # Decode the binder region tokens to an amino acid string.
            # convert_ids_to_tokens() returns a list of single-character strings
            # (e.g. ['A','C','D','E']), which we join directly into a compact sequence.
            binder_tokens = seq[binder_start:binder_end].tolist()
            aa_string = "".join(tokenizer.sequence.convert_ids_to_tokens(binder_tokens))
            generated.append(aa_string)

    return generated


# ---------------------------------------------------------------------------
# Row Processor
# ---------------------------------------------------------------------------


def run_inference_for_row(
    row: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: MimirTokenizer,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Runs inference for one input row across all requested binder lengths.

    Args:
        row: Parsed row dict from the input CSV (all types already converted).
        model: ESM-3 model in eval mode.
        tokenizer: MimirTokenizer instance.
        device: Torch device.
        seed: Random seed forwarded to generate_sequences.

    Returns:
        List of result dicts (one per generated sequence).
    """
    lmdb_path = row["lmdb_path"]
    key = row["key"]
    start_pos = row["start_pos"]
    end_pos = row["end_pos"]
    min_len = row["min_len"]
    max_len = row["max_len"]
    num_seqs_per_len = row["num_seqs_per_len"]
    unmask_per_iter = row["unmask_per_iter"]
    temperature = row["temperature"]
    checkpoint_dir = row["checkpoint_dir"]

    fingerprint = load_target_from_lmdb(lmdb_path, key, start_pos, end_pos)

    results: list[dict[str, Any]] = []

    for binder_len in range(min_len, max_len + 1):
        logger.info(f"  Generating {num_seqs_per_len} sequences of length {binder_len} for {key}")

        input_tensors = build_input_tensors(
            fingerprint=fingerprint,
            binder=None,
            tokenizer=tokenizer,
            binder_len=binder_len,
        )

        sequences = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            input_tensors=input_tensors,
            binder_len=binder_len,
            num_sequences=num_seqs_per_len,
            unmask_per_iter=unmask_per_iter,
            temperature=temperature,
            device=device,
        )

        for seq_str in sequences:
            results.append({
                "checkpoint_dir": checkpoint_dir,
                "lmdb_path": lmdb_path,
                "key": key,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "inferred_sequence": seq_str,
                "unmask_per_iter": unmask_per_iter,
                "temperature": temperature,
            })

    return results


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_INT_COLS = {"start_pos", "end_pos", "min_len", "max_len", "num_seqs_per_len", "unmask_per_iter"}
_FLOAT_COLS = {"temperature"}


def read_input_csv(path: str, sep: str = "\t") -> list[dict[str, Any]]:
    """Reads and validates the inference parameters CSV file.

    Args:
        path: Path to input TSV/CSV file.
        sep: Column delimiter character.

    Returns:
        List of row dicts with numeric types already converted.

    Raises:
        ValueError: If any row fails validation (temperature, length bounds).
    """
    rows: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for i, row in enumerate(reader, start=2):  # start=2: row 1 is header
            parsed: dict[str, Any] = {}
            for col, val in row.items():
                if col in _INT_COLS:
                    parsed[col] = int(val)
                elif col in _FLOAT_COLS:
                    parsed[col] = float(val)
                else:
                    parsed[col] = val

            # Validation
            if parsed["temperature"] <= 0:
                raise ValueError(f"Row {i}: temperature must be > 0, got {parsed['temperature']}")
            if parsed["min_len"] > parsed["max_len"]:
                raise ValueError(
                    f"Row {i}: min_len ({parsed['min_len']}) > max_len ({parsed['max_len']})"
                )
            if parsed["start_pos"] > parsed["end_pos"]:
                raise ValueError(
                    f"Row {i}: start_pos ({parsed['start_pos']}) > end_pos ({parsed['end_pos']})"
                )
            if parsed["start_pos"] < 1:
                raise ValueError(f"Row {i}: start_pos must be >= 1, got {parsed['start_pos']}")
            if parsed["unmask_per_iter"] < 1:
                raise ValueError(
                    f"Row {i}: unmask_per_iter must be >= 1, got {parsed['unmask_per_iter']}"
                )

            rows.append(parsed)

    return rows


_OUTPUT_FIELDNAMES = [
    "checkpoint_dir",
    "lmdb_path",
    "key",
    "start_pos",
    "end_pos",
    "inferred_sequence",
    "unmask_per_iter",
    "temperature",
]


def write_output_csv(path: str, results: list[dict[str, Any]], sep: str = "\t") -> None:
    """Writes generated sequences to a TSV/CSV file.

    Args:
        path: Output file path.
        results: List of result dicts from run_inference_for_row.
        sep: Column delimiter character.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDNAMES, delimiter=sep, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_inference(
    input_path: Path,
    output_path: Path,
    sep: str,
    seed: int,
) -> None:
    """Reads parameters, loads the model once, and generates all sequences.

    The checkpoint directory is taken from the first row of the input file.
    All rows must use the same checkpoint.

    Args:
        input_path: Path to the input TSV/CSV file.
        output_path: Path to write the output TSV/CSV file.
        sep: Column delimiter for both files.
        seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    rows = read_input_csv(str(input_path), sep)
    if not rows:
        logger.warning("Input file is empty — nothing to do.")
        return

    checkpoint_dir = rows[0]["checkpoint_dir"]
    logger.info(f"Loading model from checkpoint: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(checkpoint_path=checkpoint_dir)

    # CPU kernels for some operations in ESM3 don't support BFloat16 (e.g. coords).
    # We force Float32 on CPU for stability. GPU remains BFloat16 for performance.
    if device.type == "cpu":
        model = model.float()
        logger.info("Forced model to float32 for CPU compatibility")

    model.to(device)
    model.eval()

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer()

    all_results: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        key = row["key"]
        logger.info(f"Processing row {i}/{len(rows)}: key={key}")
        try:
            row_results = run_inference_for_row(row, model, tokenizer, device)
            all_results.extend(row_results)
            logger.info(f"  → {len(row_results)} sequences generated")
        except KeyError as exc:
            logger.error(f"Row {i} (key={key}): LMDB key not found — {exc}. Skipping.")
        except RuntimeError as exc:
            logger.error(f"Row {i} (key={key}): model error — {exc}. Skipping.")

    write_output_csv(str(output_path), all_results, sep)
    logger.info(
        f"Done. {len(rows)} rows processed, {len(all_results)} sequences written to {output_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the MÍMIR inference script."""
    parser = argparse.ArgumentParser(
        description="Generate peptide binders with fine-tuned ESM-3 + LoRA."
    )
    parser.add_argument("--input", required=True, help="Path to input TSV/CSV file.")
    parser.add_argument("--output", "-o", required=True, help="Path to output TSV/CSV file.")
    parser.add_argument(
        "--sep",
        default="\t",
        help='Column separator. Use "\\t" for TSV (default) or "," for CSV.',
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    run_inference(
        input_path=input_path,
        output_path=output_path,
        sep=args.sep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
Unit tests for scripts/inference.py.

All tests are designed to run without a GPU, real LMDB, or actual ESM-3 weights.
The ESM-3 forward pass is mocked with a lightweight stub.
"""

# ---------------------------------------------------------------------------
# Stdlib imports
# ---------------------------------------------------------------------------
import csv
import io
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import pytest
import torch

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from scripts.inference import (
    _OUTPUT_FIELDNAMES,
    generate_sequences,
    read_input_csv,
    write_output_csv,
)
from mimir.tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Real MimirTokenizer loaded once for the module."""
    return load_tokenizer()


def _make_mock_model(vocab_size: int = 33, struct_vocab_size: int = 4101) -> MagicMock:
    """Returns a mock ESM-3 that produces logits biased toward valid AA tokens.

    ESM3 sequence vocab: AA token IDs are 4-23 (L, A, G, V, S, E, R, T, I, D,
    P, K, Q, N, F, Y, M, H, W, C). IDs 0-3 and 24+ are special tokens that
    map to None in convert_ids_to_tokens. We mask those out so multinomial
    sampling always picks a real amino acid.
    """
    AA_START, AA_END = 4, 24  # exclusive end

    def fake_forward(**kwargs):
        seq = kwargs["sequence_tokens"]
        B, L = seq.shape
        # Start with -inf everywhere, then zero out the valid AA range so that
        # softmax concentrates probability on amino acids only.
        logits = torch.full((B, L, vocab_size), -1e9)
        logits[:, :, AA_START:AA_END] = 0.0
        out = MagicMock()
        out.sequence_logits = logits
        out.structure_logits = torch.zeros(B, L, struct_vocab_size)
        return out

    model = MagicMock(side_effect=fake_forward)
    # Mock parameters() to return an iterator, like a real nn.Module
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.side_effect = lambda: iter([dummy_param])
    return model


# ---------------------------------------------------------------------------
# CSV I/O Tests
# ---------------------------------------------------------------------------


def _write_tsv(rows: list[dict[str, Any]], header: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


_VALID_HEADER = [
    "checkpoint_dir", "lmdb_path", "key",
    "start_pos", "end_pos", "min_len", "max_len",
    "num_seqs_per_len", "unmask_per_iter", "temperature",
]

_VALID_ROW = {
    "checkpoint_dir": "runs/run2/epoch_22",
    "lmdb_path": "data/features.lmdb",
    "key": "P27986",
    "start_pos": "3",
    "end_pos": "79",
    "min_len": "8",
    "max_len": "10",
    "num_seqs_per_len": "2",
    "unmask_per_iter": "2",
    "temperature": "1.0",
}


def test_read_input_csv_valid():
    tsv = _write_tsv([_VALID_ROW], _VALID_HEADER)
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        f.write(tsv)
        tmp = f.name

    rows = read_input_csv(tmp, sep="\t")
    assert len(rows) == 1
    row = rows[0]
    assert row["key"] == "P27986"
    assert row["start_pos"] == 3
    assert row["end_pos"] == 79
    assert row["min_len"] == 8
    assert row["max_len"] == 10
    assert row["num_seqs_per_len"] == 2
    assert row["unmask_per_iter"] == 2
    assert row["temperature"] == pytest.approx(1.0)


def test_read_input_csv_invalid_temperature():
    bad_row = {**_VALID_ROW, "temperature": "0.0"}
    tsv = _write_tsv([bad_row], _VALID_HEADER)
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        f.write(tsv)
        tmp = f.name

    with pytest.raises(ValueError, match="temperature"):
        read_input_csv(tmp, sep="\t")


def test_read_input_csv_invalid_min_max_len():
    bad_row = {**_VALID_ROW, "min_len": "12", "max_len": "8"}
    tsv = _write_tsv([bad_row], _VALID_HEADER)
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        f.write(tsv)
        tmp = f.name

    with pytest.raises(ValueError, match="min_len"):
        read_input_csv(tmp, sep="\t")


def test_read_input_csv_invalid_start_pos():
    bad_row = {**_VALID_ROW, "start_pos": "0"}
    tsv = _write_tsv([bad_row], _VALID_HEADER)
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        f.write(tsv)
        tmp = f.name

    with pytest.raises(ValueError, match="start_pos"):
        read_input_csv(tmp, sep="\t")


def test_write_output_csv_roundtrip():
    results = [
        {
            "checkpoint_dir": "runs/run2/epoch_22",
            "lmdb_path": "data/features.lmdb",
            "key": "P27986",
            "start_pos": 3,
            "end_pos": 79,
            "inferred_sequence": "ACDEFGHIKL",
            "unmask_per_iter": 2,
            "temperature": 1.0,
        }
    ]

    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        tmp = f.name

    write_output_csv(tmp, results, sep="\t")

    with open(tmp, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        read_rows = list(reader)

    assert len(read_rows) == 1
    assert read_rows[0]["inferred_sequence"] == "ACDEFGHIKL"
    assert read_rows[0]["key"] == "P27986"
    assert list(read_rows[0].keys()) == _OUTPUT_FIELDNAMES


# ---------------------------------------------------------------------------
# Inference Engine Tests
# ---------------------------------------------------------------------------


def _make_input_tensors(tokenizer, fp_len: int = 5, binder_len: int = 4):
    """Builds minimal inference tensors using a synthetic fingerprint."""
    from mimir.tokenizer import build_input_tensors

    fingerprint = {
        "sequence": "ACDEF",
        "structure_tokens": [0] * fp_len,
        "sasa": [0.0] * fp_len,
        "coordinates": [[[0.0, 0.0, 0.0]] * 3] * fp_len,
    }
    return build_input_tensors(
        fingerprint=fingerprint,
        binder=None,
        tokenizer=tokenizer,
        binder_len=binder_len,
    )


def test_generate_sequences_count(tokenizer):
    """generate_sequences returns exactly num_sequences results."""
    binder_len = 4
    num_sequences = 3
    input_tensors = _make_input_tensors(tokenizer, binder_len=binder_len)
    model = _make_mock_model()
    device = torch.device("cpu")

    results = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        input_tensors=input_tensors,
        binder_len=binder_len,
        num_sequences=num_sequences,
        unmask_per_iter=2,
        temperature=1.0,
        device=device,
    )

    assert len(results) == num_sequences


def test_generate_sequences_length(tokenizer):
    """Each generated sequence should have the expected amino acid length."""
    binder_len = 6
    input_tensors = _make_input_tensors(tokenizer, binder_len=binder_len)
    model = _make_mock_model()
    device = torch.device("cpu")

    results = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        input_tensors=input_tensors,
        binder_len=binder_len,
        num_sequences=2,
        unmask_per_iter=2,
        temperature=1.0,
        device=device,
    )

    for seq in results:
        assert len(seq) == binder_len, f"Expected length {binder_len}, got '{seq}' (len={len(seq)})"


def test_generate_sequences_reproducible(tokenizer):
    """Same seed must produce identical outputs."""
    binder_len = 5
    input_tensors = _make_input_tensors(tokenizer, binder_len=binder_len)
    model = _make_mock_model()
    device = torch.device("cpu")

    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        input_tensors=input_tensors,
        binder_len=binder_len,
        num_sequences=2,
        unmask_per_iter=1,
        temperature=1.0,
        device=device,
    )

    torch.manual_seed(999)
    result_a = generate_sequences(**kwargs)
    torch.manual_seed(999)
    result_b = generate_sequences(**kwargs)

    assert result_a == result_b


def test_generate_sequences_invalid_temperature(tokenizer):
    """Temperature <= 0 should raise ValueError."""
    input_tensors = _make_input_tensors(tokenizer, binder_len=4)
    model = _make_mock_model()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="temperature"):
        generate_sequences(
            model=model,
            tokenizer=tokenizer,
            input_tensors=input_tensors,
            binder_len=4,
            num_sequences=1,
            unmask_per_iter=1,
            temperature=0.0,
            device=device,
        )

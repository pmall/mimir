"""
Structure feature extraction for PDB/mmCIF files.

Provides utilities to extract 3-track features (sequence, coordinates, SASA)
from protein structures for use with ESM-3.
"""

import io
import logging
import warnings
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import zstandard as zstd
from esm.utils.structure.protein_chain import ProteinChain

logger = logging.getLogger(__name__)


def parse_mmcif(cif_content: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse mmCIF content and extract 3-track features.

    Args:
        cif_content: Raw mmCIF file content as string.

    Returns:
        Tuple of (sequence, coordinates, sasa):
            - sequence: str, amino acid sequence (1-letter codes)
            - coordinates: np.ndarray, shape (L, 37, 3), atom37 representation
            - sasa: np.ndarray, shape (L,), per-residue SASA values

    Raises:
        ValueError: If structure cannot be parsed or contains non-standard residues.
    """
    try:
        cif_file = pdbx.CIFFile.read(io.StringIO(cif_content))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*auth_atom_id.*", category=UserWarning
            )
            structure = pdbx.get_structure(cif_file, model=1)
        structure = structure[struc.filter_amino_acids(structure)]

        # Drop residues with insertion codes
        if hasattr(structure, "ins_code"):
            mask = np.isin(structure.ins_code, ["", " "])
            structure = structure[mask]

        # Remap multi-character chain IDs to single characters
        unique_chains = list(dict.fromkeys(structure.chain_id))
        if any(len(c) > 1 for c in unique_chains):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            if len(unique_chains) > len(alphabet):
                raise ValueError(
                    f"Too many chains ({len(unique_chains)}), "
                    f"exceeds remapping capacity ({len(alphabet)})"
                )
            remap = {old: alphabet[i] for i, old in enumerate(unique_chains)}
            structure = structure.copy()
            structure.chain_id = np.array(
                [remap[c] for c in structure.chain_id], dtype="U1"
            )

        chain = ProteinChain.from_atomarray(structure)

        # Extract sequence
        sequence = chain.sequence

        # Extract coordinates (atom37)
        params = chain.to_structure_encoder_inputs()
        coords = params[0]
        if hasattr(coords, "numpy"):
            coords = coords.numpy()
        if coords.ndim == 4 and coords.shape[0] == 1:
            coords = coords[0]  # Remove batch dim

        # Extract SASA
        sasa = chain.sasa()
        if hasattr(sasa, "numpy"):
            sasa = sasa.numpy()

        return sequence, coords, sasa

    except Exception as e:
        raise ValueError(f"Failed to parse structure: {e}") from e


def parse_mmcif_bytes(cif_bytes: bytes, compressed: bool = False) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse mmCIF from bytes with optional decompression.

    Args:
        cif_bytes: Raw bytes of mmCIF file content.
        compressed: If True, decompress using zstandard first.

    Returns:
        Tuple of (sequence, coordinates, sasa). See parse_mmcif().

    Raises:
        ValueError: If decompression or parsing fails.
    """
    try:
        if compressed:
            dctx = zstd.ZstdDecompressor()
            cif_content = dctx.decompress(cif_bytes).decode("utf-8")
        else:
            cif_content = cif_bytes.decode("utf-8")

        return parse_mmcif(cif_content)

    except Exception as e:
        raise ValueError(f"Failed to parse mmCIF bytes: {e}") from e


def parse_mmcif_file(file_path: Path, compressed: bool = False) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse mmCIF from file.

    Args:
        file_path: Path to the mmCIF file.
        compressed: If True, file is zstd compressed.

    Returns:
        Tuple of (sequence, coordinates, sasa). See parse_mmcif().

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If parsing fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"mmCIF file not found: {file_path}")

    with open(file_path, "rb") as f:
        content = f.read()

    return parse_mmcif_bytes(content, compressed=compressed)


class StructureFeatures:
    """Container for structure features with serialization support."""

    def __init__(
        self,
        entry_id: str,
        target: str,
        sequence: str,
        coordinates: np.ndarray | None = None,
        sasa: np.ndarray | None = None,
    ):
        """Initialize structure features.

        Args:
            entry_id: Unique identifier for this entry.
            target: Target protein accession (e.g., UniProt ID).
            sequence: Amino acid sequence (1-letter codes).
            coordinates: Atom37 coordinates, shape (L, 37, 3) or None.
            sasa: Per-residue SASA values, shape (L,) or None.
        """
        self.entry_id = entry_id
        self.target = target
        self.sequence = sequence
        self.coordinates = coordinates
        self.sasa = sasa

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with keys: id, target, sequence, coordinates, sasa
        """
        return {
            "id": self.entry_id,
            "target": self.target,
            "sequence": self.sequence,
            "coordinates": (
                self.coordinates.tolist()
                if isinstance(self.coordinates, np.ndarray)
                else None
            ),
            "sasa": (
                self.sasa.tolist()
                if isinstance(self.sasa, np.ndarray)
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureFeatures":
        """Create from dictionary.

        Args:
            data: Dictionary with keys id, target, sequence, coordinates, sasa.

        Returns:
            StructureFeatures instance.
        """
        coords = data.get("coordinates")
        sasa = data.get("sasa")

        if coords is not None:
            coords = np.array(coords)
        if sasa is not None:
            sasa = np.array(sasa)

        return cls(
            entry_id=data["id"],
            target=data["target"],
            sequence=data["sequence"],
            coordinates=coords,
            sasa=sasa,
        )

    def has_structure(self) -> bool:
        """Check if structure features are available."""
        return self.coordinates is not None and self.sasa is not None


__all__ = [
    "parse_mmcif",
    "parse_mmcif_bytes",
    "parse_mmcif_file",
    "StructureFeatures",
]

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


class ParsedStructure:
    """Standard parsed structure results."""
    def __init__(self, sequence: str, coords: np.ndarray, sasa: np.ndarray):
        self.sequence = sequence
        self.coords = coords
        self.sasa = sasa


class ParsedTargetStructure(ParsedStructure):
    """Parsed target structure results including pLDDT factors."""
    def __init__(
        self,
        sequence: str,
        coords: np.ndarray,
        sasa: np.ndarray,
        global_plddt: float,
        residue_plddt: np.ndarray,
    ):
        super().__init__(sequence, coords, sasa)
        self.global_plddt = global_plddt
        self.residue_plddt = residue_plddt


def _get_clean_atom_array(
    cif_content: str,
    chain_id: str | None = None,
    extra_fields: list[str] | None = None
) -> bs.AtomArray if 'bs' in globals() else struc.AtomArray:
    """Parse mmCIF and return a clean Biotite AtomArray filtered for standard amino acids.

    Filters out hetero-atoms, insertions, and targets a specific chain id if provided.

    Raises:
        ValueError: If structure cannot be parsed or contains no valid residues.
    """
    try:
        cif_file = pdbx.CIFFile.read(io.StringIO(cif_content))
        
        fields = ["auth_asym_id"]
        if extra_fields:
            fields.extend(extra_fields)
            
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*auth_atom_id.*", category=UserWarning
            )
            structure = pdbx.get_structure(cif_file, model=1, extra_fields=fields)
            
        structure = structure[struc.filter_amino_acids(structure)]

        # Filter out hetero atoms (HETATM records)
        structure = structure[~structure.hetero]

        # Drop residues with insertion codes
        if hasattr(structure, "ins_code"):
            mask = np.isin(structure.ins_code, ["", " "])
            structure = structure[mask]

        if len(structure) == 0:
            raise ValueError("No standard residues remain after filtering")

        if chain_id:
            # Filter specifically for the requested chain
            chain_mask = None
            if hasattr(structure, "auth_asym_id"):
                chain_mask = structure.auth_asym_id == chain_id
            
            if chain_mask is None or not chain_mask.any():
                # Fallback to label_asym_id (chain_id)
                chain_mask = structure.chain_id == chain_id
                
            if not chain_mask.any():
                raise ValueError(
                    f"Chain {chain_id} has no standard residues after filtering"
                )
            structure = structure[chain_mask]

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

        return structure

    except Exception as e:
        raise ValueError(f"Failed to parse structure: {e}") from e


def _extract_3_tracks(structure: struc.AtomArray) -> tuple[str, np.ndarray, np.ndarray]:
    """Given a clean AtomArray, use ProteinChain to extract seq, coords, and sasa."""
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


def parse_mmcif(cif_content: str, chain_id: str | None = None) -> ParsedStructure:
    """Parse mmCIF content and extract standard 3-track features.

    Args:
        cif_content: Raw mmCIF file content as string.
        chain_id: Optional chain ID to filter by. If None, uses all chains.

    Returns:
        A ParsedStructure object containing sequence, coordinates, and sasa.

    Raises:
        ValueError: If structure cannot be parsed or contains non-standard residues.
    """
    structure = _get_clean_atom_array(cif_content, chain_id)
    sequence, coords, sasa = _extract_3_tracks(structure)
    return ParsedStructure(sequence=sequence, coords=coords, sasa=sasa)


def parse_af2_mmcif(cif_content: str, chain_id: str | None = None) -> ParsedTargetStructure:
    """Parse an AlphaFold2 mmCIF file, extracting features, per-residue pLDDT, and global pLDDT.

    Args:
        cif_content: Raw mmCIF file content as a string.
        chain_id: Optional chain ID to filter by.

    Returns:
        A ParsedTargetStructure containing sequence, coords, sasa, global plddt, and residue plddts.

    Raises:
        ValueError: If parsing fails or metrics are not found.
    """
    # 1. Regex fast search for global pLDDT
    import re
    match = re.search(r"_ma_qa_metric_global\.metric_value\s+([\d\.]+)", cif_content)
    if not match:
        raise ValueError("Could not find global pLDDT (_ma_qa_metric_global.metric_value) in AF2 mmCIF.")
    
    try:
        global_plddt = float(match.group(1))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse global pLDDT value: {e}") from e

    # 2. Extract strictly B-factors alongside standard tracking
    structure = _get_clean_atom_array(cif_content, chain_id, extra_fields=["b_factor"])
    
    # 3. Pull per-residue B-factor corresponding precisely to CA atoms
    ca_atoms = structure[structure.atom_name == "CA"]
    if not hasattr(ca_atoms, "b_factor"):
        raise ValueError("Failed to retrieve b_factor column from AF2 structure.")
    residue_plddt = ca_atoms.b_factor
    
    # 4. Extract seq, coords, sasa natively
    sequence, coords, sasa = _extract_3_tracks(structure)
    
    if len(sequence) != len(residue_plddt):
        raise ValueError(
            f"Extraction mismatch: {len(sequence)} amino acids but {len(residue_plddt)} CA B-factors."
        )

    return ParsedTargetStructure(
        sequence=sequence,
        coords=coords,
        sasa=sasa,
        global_plddt=global_plddt,
        residue_plddt=residue_plddt,
    )


def parse_mmcif_bytes(cif_bytes: bytes, compressed: bool = False, chain_id: str | None = None) -> ParsedStructure:
    try:
        if compressed:
            dctx = zstd.ZstdDecompressor()
            cif_content = dctx.decompress(cif_bytes).decode("utf-8")
        else:
            cif_content = cif_bytes.decode("utf-8")

        return parse_mmcif(cif_content, chain_id=chain_id)
    except Exception as e:
        raise ValueError(f"Failed to parse mmCIF bytes: {e}") from e


def parse_af2_mmcif_bytes(
    cif_bytes: bytes, compressed: bool = False, chain_id: str | None = None
) -> ParsedTargetStructure:
    try:
        if compressed:
            dctx = zstd.ZstdDecompressor()
            cif_content = dctx.decompress(cif_bytes).decode("utf-8")
        else:
            cif_content = cif_bytes.decode("utf-8")

        return parse_af2_mmcif(cif_content, chain_id=chain_id)
    except Exception as e:
        raise ValueError(f"Failed to parse AF2 mmCIF bytes: {e}") from e


def parse_mmcif_file(file_path: Path, compressed: bool = False, chain_id: str | None = None) -> ParsedStructure:
    if not file_path.exists():
        raise FileNotFoundError(f"mmCIF file not found: {file_path}")

    with open(file_path, "rb") as f:
        content = f.read()

    return parse_mmcif_bytes(content, compressed=compressed, chain_id=chain_id)


MIN_FINGERPRINT_LEN = 15

# Tien et al. 2013 Maximum allowed SASA (Theoretical max in Gly-X-Gly). Used for rSASA.
MAX_SASA_REFERENCE = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
}


def compute_rsasa(sequence: str, sasa: list[float] | np.ndarray) -> np.ndarray:
    """Compute Relative SASA arrays safely. Handle unknown residues by assuming minimum surface."""
    rsasa = np.zeros(len(sequence), dtype=np.float32)
    for i, (res, abs_sasa) in enumerate(zip(sequence, sasa)):
        max_sasa = MAX_SASA_REFERENCE.get(res.upper(), 1.0)  # avoid division by 0
        rsasa[i] = abs_sasa / max_sasa
    return rsasa


def get_fingerprint_mask(
    sequence: str,
    sasa: list[float] | np.ndarray,
    plddt: list[float] | np.ndarray,
    max_len: int = 157,
) -> np.ndarray | None:
    """Returns a boolean mask of the kept positions, or None if skipped by min-length.
    
    Filters positions based on pLDDT >= 70.0 and rSASA >= 0.15.
    If valid positions > max_len, keeps only the highest rSASA positions sequentially.
    """
    rsasa_np = compute_rsasa(sequence, sasa)
    plddt_np = np.array(plddt)
    
    # Base masking rules
    mask = (plddt_np >= 70.0) & (rsasa_np >= 0.15)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) < MIN_FINGERPRINT_LEN:
        return None
        
    if len(valid_indices) > max_len:
        # Get rsasa values for the currently valid indices only
        valid_rsasa = rsasa_np[valid_indices]
        
        # Find the indices of the top `max_len` values within the valid subset
        top_subset_indices = np.argsort(valid_rsasa)[-max_len:]
        
        # Map those subset indices back to the original chronological sequence indices
        final_indices = valid_indices[top_subset_indices]
        
        # Create a strict mask keeping only those final selected indices
        strict_mask = np.zeros_like(mask, dtype=bool)
        strict_mask[final_indices] = True
        return strict_mask
        
    return mask


class BinderFeatures:
    """Container for binder features with serialization support."""

    def __init__(
        self,
        entry_id: str,
        sequence: str,
        structure_tokens: list[int] | None = None,
        sasa: list[float] | None = None,
    ):
        self.entry_id = entry_id
        self.sequence = sequence
        self.structure_tokens = structure_tokens
        self.sasa = sasa

    def to_dict(self) -> dict:
        return {
            "id": self.entry_id,
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
            "sasa": self.sasa,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BinderFeatures":
        return cls(
            entry_id=data["id"],
            sequence=data["sequence"],
            structure_tokens=data.get("structure_tokens"),
            sasa=data.get("sasa"),
        )

    def has_structure(self) -> bool:
        return self.structure_tokens is not None and self.sasa is not None


class TargetFeatures:
    """Container for target features with strict initialization and serialization support."""

    def __init__(
        self,
        entry_id: str,
        sequence: str,
        structure_tokens: list[int],
        sasa: list[float],
        plddt: float,
        residue_plddt: list[float],
        position_ids: list[int],
    ):
        self.entry_id = entry_id
        self.sequence = sequence
        self.structure_tokens = structure_tokens
        self.sasa = sasa
        self.plddt = plddt
        self.residue_plddt = residue_plddt
        self.position_ids = position_ids

    def to_dict(self) -> dict:
        return {
            "id": self.entry_id,
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
            "sasa": self.sasa,
            "plddt": self.plddt,
            "residue_plddt": self.residue_plddt,
            "position_ids": self.position_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TargetFeatures":
        return cls(
            entry_id=data.get("id", ""),
            sequence=data.get("sequence", ""),
            structure_tokens=data.get("structure_tokens", []),
            sasa=data.get("sasa", []),
            plddt=data.get("plddt", 0.0),
            residue_plddt=data.get("residue_plddt", []),
            position_ids=data.get("position_ids", []),
        )


class FingerprintFeatures:
    """Container for condensed fingerprint features returned by the fingerprinting model."""

    def __init__(
        self,
        entry_id: str,
        sequence: str,
        structure_tokens: list[int],
        sasa: list[float],
        residue_plddt: list[float],
        position_ids: list[int],
    ):
        self.entry_id = entry_id
        self.sequence = sequence
        self.structure_tokens = structure_tokens
        self.sasa = sasa
        self.residue_plddt = residue_plddt
        self.position_ids = position_ids

    def to_dict(self) -> dict:
        return {
            "id": self.entry_id,
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
            "sasa": self.sasa,
            "residue_plddt": self.residue_plddt,
            "position_ids": self.position_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FingerprintFeatures":
        return cls(
            entry_id=data.get("id", ""),
            sequence=data.get("sequence", ""),
            structure_tokens=data.get("structure_tokens", []),
            sasa=data.get("sasa", []),
            residue_plddt=data.get("residue_plddt", []),
            position_ids=data.get("position_ids", []),
        )


__all__ = [
    "ParsedStructure",
    "ParsedTargetStructure",
    "parse_mmcif",
    "parse_mmcif_bytes",
    "parse_mmcif_file",
    "parse_af2_mmcif",
    "parse_af2_mmcif_bytes",
    "BinderFeatures",
    "TargetFeatures",
    "FingerprintFeatures",
    "compute_rsasa",
    "get_fingerprint_mask",
    "MIN_FINGERPRINT_LEN",
    "MAX_SASA_REFERENCE",
]

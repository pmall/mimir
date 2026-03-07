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
        
        fields = ["auth_asym_id", "label_seq_id"]
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
        coords = coords[0, :, :3, :]  # Remove batch dim and keep only backbone N, CA, C

    # Extract SASA
    sasa = chain.sasa()
    if hasattr(sasa, "numpy"):
        sasa = sasa.numpy()

    return sequence, coords, sasa
class ParsedBinderStructure:
    """Standard parsed structure results for Binders (no SASA)."""
    def __init__(self, sequence: str, coords: np.ndarray):
        self.sequence = sequence
        self.coords = coords



def parse_binder_mmcif(cif_content: str, reference_sequence: str, chain_id: str | None = None) -> ParsedBinderStructure | None:
    """Parse mmCIF content for binders and align structure to a reference sequence.

    Args:
        cif_content: Raw mmCIF file content as string.
        reference_sequence: Biological sequence of the binder (required for alignment).
        chain_id: Optional chain ID to filter by. If None, uses all chains.

    Returns:
        A ParsedBinderStructure object containing the sequence and aligned coordinates,
        or None if no structural atoms remain after filtering.

    Raises:
        ValueError: If structure cannot be parsed.
    """
    try:
        structure = _get_clean_atom_array(cif_content, chain_id)
    except ValueError as e:
        if "no standard residues" in str(e).lower():
            return None
        raise e

    # We only take the coordinate extraction logic to map the structure correctly
    chain_encoder = ProteinChain.from_atomarray(structure)
    params = chain_encoder.to_structure_encoder_inputs()
    coords = params[0]
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]  # Remove batch dim -> (L_resolved, 37, 3)
        
    # The atom array gives us the label_seq_id mapping
    # Note: Biotite's from_atomarray uses contiguous indices internally for the returned shape,
    # but we can query the original structure CA atoms to map biological indices.
    ca_atoms = structure[structure.atom_name == "CA"]
    
    if len(ca_atoms) != coords.shape[0]:
        # Fallback if mapping diverges
        raise ValueError("Mismatch between CA atoms and extracted coordinates.")

    L = len(reference_sequence)
    aligned_coords = np.full((L, 37, 3), np.nan, dtype=np.float32)
    
    # Place each resolved residue at its biological index 
    # (label_seq_id is 1-indexed, biological index is 0-indexed)
    valid_mapping = False
    for i, seq_id in enumerate(ca_atoms.label_seq_id):
        try:
            bio_idx = int(seq_id) - 1
        except ValueError:
            continue
            
        if 0 <= bio_idx < L:
            aligned_coords[bio_idx] = coords[i]
            valid_mapping = True
            
    if not valid_mapping:
         return None

    return ParsedBinderStructure(sequence=reference_sequence, coords=aligned_coords)


def parse_binder_mmcif_bytes(cif_bytes: bytes, reference_sequence: str, compressed: bool = False, chain_id: str | None = None) -> ParsedBinderStructure | None:
    try:
        if compressed:
            dctx = zstd.ZstdDecompressor()
            cif_content = dctx.decompress(cif_bytes).decode("utf-8")
        else:
            cif_content = cif_bytes.decode("utf-8")

        return parse_binder_mmcif(cif_content, reference_sequence=reference_sequence, chain_id=chain_id)
    except Exception as e:
        raise ValueError(f"Failed to parse mmCIF bytes: {e}") from e


def parse_af2_mmcif(cif_content: str, chain_id: str | None = None) -> ParsedTargetStructure:
    """Parse mmCIF content from AlphaFold2 and extract sequences, coords, SASA, and pLDDT.

    Args:
        cif_content: Raw mmCIF file content as string.
        chain_id: Optional chain ID to filter by. If None, uses all chains.

    Returns:
        A ParsedTargetStructure object containing global and per-residue pLDDT,
        or None if no structural atoms remain after filtering.

    Raises:
        ValueError: If structure cannot be parsed.
    """
    structure = _get_clean_atom_array(cif_content, chain_id, extra_fields=["B_iso_or_equiv"])
    
    sequence, coords, sasa = _extract_3_tracks(structure)

    # Extract pLDDT from B-factor column of CA atoms
    ca_atoms = structure[structure.atom_name == "CA"]
    
    if len(ca_atoms) != coords.shape[0]:
        raise ValueError("Mismatch between CA atoms and extracted coordinates.")

    residue_plddt = ca_atoms.B_iso_or_equiv.astype(np.float32)
    global_plddt = float(np.mean(residue_plddt))

    return ParsedTargetStructure(
        sequence=sequence,
        coords=coords,
        sasa=sasa,
        global_plddt=global_plddt,
        residue_plddt=residue_plddt,
    )


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


def get_smoothed_rsasa(rsasa_np: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Compute smoothed rSASA using a sliding window.
    
    Properly handles boundary edges by averaging only over available amino acids.
    """
    half_window = window_size // 2
    n = len(rsasa_np)
    
    smoothed_rsasa = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # Window bounds in original array space
        start_idx = max(0, i - half_window)
        end_idx = min(n - 1, i + half_window)
        smoothed_rsasa[i] = np.mean(rsasa_np[start_idx:end_idx + 1])
        
    return smoothed_rsasa


def get_fingerprint_mask(
    sequence: str,
    sasa: list[float] | np.ndarray,
    plddt: list[float] | np.ndarray,
    max_len: int = 157,
) -> tuple[np.ndarray | None, float | None]:
    """Returns a boolean mask of the kept positions and the applied rSASA threshold.
    
    Returns (mask, threshold) or (None, None) if skipped by min-length.
    Filters by pLDDT >= 70.0. If valid positions > max_len, incrementally 
    filters by smoothed rSASA (window size 15) using a 0.01 threshold 
    increment until it fits within max_len.
    """
    rsasa_np = compute_rsasa(sequence, sasa)
    plddt_np = np.array(plddt)
    
    # Compute smoothed rSASA (sliding window of 15: 7 before, 7 after)
    smoothed_rsasa = get_smoothed_rsasa(rsasa_np, window_size=15)
        
    # Base masking rules: strictly pLDDT >= 70.0
    base_mask = plddt_np >= 70.0
    valid_indices = np.where(base_mask)[0]
    
    # The min length filter applies after plddt filtering
    if len(valid_indices) < MIN_FINGERPRINT_LEN:
        return None, None
        
    if len(valid_indices) <= max_len:
        return base_mask, None
        
    # Iterative step-wise thresholding on smoothed rSASA
    threshold = 0.01
    
    while True:
        current_mask = base_mask & (smoothed_rsasa >= threshold)
        
        if current_mask.sum() <= max_len:
            # We reached the target length budget
            return current_mask, round(threshold, 2)
            
        # If we accidentally cull too much, fail gracefully
        if current_mask.sum() < MIN_FINGERPRINT_LEN:
            return None, None
            
        threshold += 0.01


class ParsedBinderStructure:
    """Standard parsed structure results for Binders (no SASA)."""
    def __init__(self, sequence: str, coords: np.ndarray):
        self.sequence = sequence
        self.coords = coords


class BinderFeatures:
    """Container for binder features with serialization support."""

    def __init__(
        self,
        entry_id: str,
        sequence: str,
        structure_tokens: list[int] | None = None,
    ):
        self.entry_id = entry_id
        self.sequence = sequence
        self.structure_tokens = structure_tokens

    def to_dict(self) -> dict:
        return {
            "id": self.entry_id,
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BinderFeatures":
        return cls(
            entry_id=data["id"],
            sequence=data["sequence"],
            structure_tokens=data.get("structure_tokens"),
        )

    def has_structure(self) -> bool:
        return self.structure_tokens is not None


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
        coordinates: list[list[list[float]]],
    ):
        self.entry_id = entry_id
        self.sequence = sequence
        self.structure_tokens = structure_tokens
        self.sasa = sasa
        self.plddt = plddt
        self.residue_plddt = residue_plddt
        self.coordinates = coordinates

    def to_dict(self) -> dict:
        return {
            "id": self.entry_id,
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
            "sasa": self.sasa,
            "plddt": self.plddt,
            "residue_plddt": self.residue_plddt,
            "coordinates": self.coordinates,
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
            coordinates=data.get("coordinates", []),
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
        coordinates: list[list[list[float]]],
        rsasa_threshold: float | None = None,
    ):
        self.entry_id = entry_id
        self.sequence = sequence
        self.structure_tokens = structure_tokens
        self.sasa = sasa
        self.residue_plddt = residue_plddt
        self.position_ids = position_ids
        self.coordinates = coordinates
        self.rsasa_threshold = rsasa_threshold

    def to_dict(self) -> dict:
        return {
            "id": self.entry_id,
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
            "sasa": self.sasa,
            "residue_plddt": self.residue_plddt,
            "position_ids": self.position_ids,
            "coordinates": self.coordinates,
            "rsasa_threshold": self.rsasa_threshold,
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
            coordinates=data.get("coordinates", []),
            rsasa_threshold=data.get("rsasa_threshold"),
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
    "get_smoothed_rsasa",
    "get_fingerprint_mask",
    "MIN_FINGERPRINT_LEN",
    "MAX_SASA_REFERENCE",
]

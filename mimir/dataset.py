"""
PyTorch Dataset for peptide sequences.

This module handles variable-length peptides (4-20 amino acids) with two strategies:
1. Static padding: Pad all sequences to max_length (20) - simple but wastes compute
2. Dynamic padding: Pad to longest sequence in batch - efficient GPU utilization

The dynamic_collate_fn implements strategy #2, which is recommended for training.
"""

import csv
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from .tokenizer import AminoAcidTokenizer


def create_dynamic_collate_fn(pad_idx: int = 0) -> Callable:
    """
    Create a collate function that pads sequences dynamically to batch-max length.
    
    Dynamic padding significantly improves training efficiency by checking
    the longest sequence in *encoded* batch and only padding to that length,
    rather than a global maximum.
    
    Args:
        pad_idx: The token index for PAD (default 0, matching tokenizer.pad_idx).
                 This is used to construct the 'pad_mask'.
    
    Returns:
        A collate_fn compatible with torch.utils.data.DataLoader
    """
    def dynamic_collate_fn(batch: list[dict]) -> dict:
        """
        Collate samples with dynamic padding to the longest sequence in batch.
        
        Process:
        1. stack lengths of all items in batch.
        2. find max_len in this batch.
        3. trim all 'tokens' tensors to max_len.
        4. create a padding mask.
        
        Input batch items have:
            - tokens: [max_length] pre-padded to global max (20).
            - length: scalar, actual sequence length.
            - target_id: scalar, target protein index or -1.
        
        Output dict has:
            - tokens: [batch, L_max] trimmed to batch-max length.
            - length: [batch] original lengths.
            - target_id: [batch] target protein indices.
            - pad_mask: [batch, L_max] boolean mask, True = PAD position.
        """
        # Extract lengths and find batch maximum
        lengths = torch.stack([item["length"] for item in batch])  # [batch]
        max_len = lengths.max().item()
        
        # Trim tokens to L_max (they were pre-padded to 20)
        tokens = torch.stack([item["tokens"][:max_len] for item in batch])  # [batch, L_max]
        
        # Build padding mask for attention: True where position >= actual length
        positions = torch.arange(max_len).unsqueeze(0)  # [1, L_max]
        pad_mask = positions >= lengths.unsqueeze(1)    # [batch, L_max] broadcast
        
        return {
            "tokens": tokens,
            "length": lengths,
            "target_id": torch.stack([item["target_id"] for item in batch]),
            "pad_mask": pad_mask,
        }
    
    return dynamic_collate_fn


class PeptideDataset(Dataset):
    """
    Dataset for peptide sequences with optional targets.

    Each sample contains:
        - sequence: Tokenized peptide sequence (padded to max_length)
        - length: Original sequence length (before padding)
        - target_id: Integer ID of target protein (-1 if no target)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer: AminoAcidTokenizer,
        max_length: int = 20,
    ):
        """
        Load dataset from CSV file.

        Args:
            dataset_path: Path to dataset.csv
            tokenizer: AminoAcidTokenizer instance
            max_length: Maximum sequence length for padding
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.target_to_id = {}

        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = row["target"]
                if target and target not in self.target_to_id:
                    self.target_to_id[target] = len(self.target_to_id)

                self.samples.append(
                    {
                        "sequence": row["sequence"],
                        "target": target,
                    }
                )

        self.id_to_target = {v: k for k, v in self.target_to_id.items()}
        self.num_targets = len(self.target_to_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sequence = sample["sequence"]

        tokens = self.tokenizer.encode(sequence, self.max_length)

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "length": torch.tensor(len(sequence), dtype=torch.long),
            "target_id": torch.tensor(
                self.target_to_id.get(sample["target"], -1), dtype=torch.long
            ),
        }

"""
PyTorch Dataset for peptide sequences.

This module handles variable-length peptides (e.g. 4-512+ amino acids) using dynamic padding.
It delegates padding to the `dynamic_collate_fn` to ensure efficient GPU utilization 
by padding only to the longest sequence in the batch.
"""

import csv
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from .tokenizer import AminoAcidTokenizer


def create_dynamic_collate_fn(pad_idx: int, mask_idx: int) -> Callable:
    """
    Create a collate function that handles dynamic padding AND masked language modeling (MLM).
    
    This function prepares batches for BERT-style training:
    1.  **Dynamic Padding**: Pads all sequences to the length of the longest sequence in the batch.
    2.  **Masking**: randomly masks ~15% of the *sequence* tokens (not targets/specials).
    3.  **Label Creation**: Creates labels for loss calculation, ignoring unmasked tokens.
    
    Args:
        pad_idx (int): The token index for PAD.
        mask_idx (int): The token index for MASK (used to replace input tokens).
    
    Returns:
        A collate_fn compatible with torch.utils.data.DataLoader
    """
    def dynamic_collate_fn(batch: list[dict]) -> dict:
        """
        Collates samples into a batch with MLM masking.
        
        Input Schema (per item):
            - tokens: Tensor[Long] of shape (L_seq) - [Target, BOS, seq..., EOS, Pad...]
            - length: int - Valid user specified length
            - target_id: int - The prepended target token
            
        Output Schema (Batched):
            - tokens: [Batch, L_max] - Masked inputs
            - labels: [Batch, L_max] - Ground truth for masked positions (-100 elsewhere)
            - pad_mask: [Batch, L_max] - True where padded
        """
        # 1. Stack and Pad
        # ----------------
        # Extract tokens and lengths
        tokens_list = [item["tokens"] for item in batch]
        lengths = torch.stack([item["length"] for item in batch])
        max_len = lengths.max().item()
        
        # Pad sequences to the max length in this batch
        from torch.nn.utils.rnn import pad_sequence
        batch_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=pad_idx)
        
        # Warning: pad_sequence pads to the longest in batch, which matches max_len
        # (Assuming no truncation happened before that violated max_length constraint relative to batch)
        
        # 2. Create Masks
        # ---------------
        # We need two masks:
        # a) padding_mask: to ignore padding in attention (and loss)
        # b) mlm_mask: to select which tokens to mask for prediction
        
        # Create position grid [1, L_max]
        positions = torch.arange(max_len, device=batch_tokens.device).unsqueeze(0)
        
        # Padding Mask: True where position >= actual length
        # Used by attention mechanism to ignore pads
        pad_mask = positions >= lengths.unsqueeze(1) 
        
        # 3. Apply BERT-style Masking
        # ---------------------------
        # We assume structure: [Target (0), BOS (1), Seq_Start (2) ... Seq_End (Len-2), EOS (Len-1)]
        # We ONLY want to mask the actual Sequence tokens.
        
        # Create a copy for labels (ground truth)
        labels = batch_tokens.clone()
        
        # Initialize inputs as a copy of ground truth (we will overwrite some with <mask>)
        inputs = batch_tokens.clone()
        
        # Iterate over each sequence in the batch to apply random masking
        # Vectorizing this fully is complex due to variable lengths, so we loop over batch (B is small ~32-64)
        batch_size = batch_tokens.size(0)
        
        for i in range(batch_size):
            # Get valid length for this sequence
            l = lengths[i].item()
            
            # Define valid range for masking:
            # Start: 2 (Skip Target @ 0 and BOS @ 1)
            # End: l - 1 (Skip EOS @ l-1)
            # Example: [T, BOS, A, B, C, EOS] (len=6) -> valid indices [2, 3, 4] (A, B, C)
            valid_start = 2
            valid_end = l - 1
            
            if valid_end <= valid_start:
                # Sequence too short to have maskable tokens (e.g., just special tokens)
                # Should not happen with valid peptides >= 4aa
                labels[i] = -100 # Ignore whole sequence
                continue
                
            # Calculate number of tokens to mask
            # Constraint: Variable masking between 25% and 75%
            # We sample a ratio uniformly from [0.25, 0.75]
            ratio = torch.empty(1).uniform_(0.25, 0.75).item()
            
            seq_len = valid_end - valid_start
            num_mask = max(1, int(seq_len * ratio))
            
            # Select random indices within the valid range
            # torch.randperm returns random permutation of 0..seq_len-1
            # We take first 'num_mask' and shift by 'valid_start'
            mask_indices = torch.randperm(seq_len)[:num_mask] + valid_start
            
            # Apply Masking to INPUTS
            inputs[i, mask_indices] = mask_idx
            
            # Set LABELS
            # We want to calculate loss ONLY on mask_indices.
            # So we set everything else to -100 (PyTorch CrossEntropy ignore_index)
            # First, set the whole label row to -100
            labels[i] = -100 
            # Then restore the ground truth ONLY at mask_indices
            labels[i, mask_indices] = batch_tokens[i, mask_indices]
            
        return {
            "tokens": inputs,      # Masked input tokens
            "labels": labels,      # -100 everywhere except masked positions
            "length": lengths,     # Original lengths
            "target_id": torch.stack([item["target_id"] for item in batch]),
            "pad_mask": pad_mask,  # Attention mask
        }
    
    return dynamic_collate_fn


class PeptideDataset(Dataset):
    """
    Dataset for peptide sequences with optional targets.

    Each sample contains:
        - sequence: Tokenized peptide sequence (No padding at dataset level; handled by collate_fn)
        - length: Original sequence length
        - target_id: Integer ID of target protein (-1 if no target)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer: AminoAcidTokenizer,
    ):
        """
        Load dataset from CSV file.

        Args:
            dataset_path: Path to dataset.csv
            tokenizer: AminoAcidTokenizer instance
        """
        self.tokenizer = tokenizer
        self.samples = []
        
        # We no longer build target_to_id here. 
        # The tokenizer must already be initialized with all targets.

        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(
                    {
                        "sequence": row["sequence"],
                        "target": row["target"],
                    }
                )
        
        # Pre-calculate lengths for LengthGroupedSampler
        self.lengths = []
        for s in self.samples:
            # +3 for Target, BOS, EOS
            l = len(s["sequence"]) + 3
            self.lengths.append(l)
            
        print(f"Dataset loaded with {len(self.samples)} samples. Max length detected: {max(self.lengths) if self.lengths else 0}")

    def get_lengths(self) -> list[int]:
        """Return list of lengths for all samples."""
        return self.lengths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sequence = sample["sequence"]
        target = sample["target"]
        
        # Get target ID from tokenizer
        try:
            target_id = self.tokenizer.get_target_id(target)
        except ValueError:
            target_id = -1 

        # Encode sequence (BOS...EOS)
        seq_tokens = self.tokenizer.encode(sequence, max_length=None)
        
        # Prepend target token
        # [TARGET] [BOS] ... [EOS]
        tokens = [target_id] + seq_tokens
        
        # No truncation logic - Return exactly what the dataset provides.
            
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "length": torch.tensor(len(tokens), dtype=torch.long),
            "target_id": torch.tensor(target_id, dtype=torch.long),
        }

"""
Tokenizer and dataloader utilities for Mimir v2.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import numpy as np
from esm.tokenization import get_esm3_model_tokenizers

# --- Constants for ESM3 Tracks ---
# Sequence vocab is 64 tokens explicitly defined.
CUT_TOKEN_ID_SEQ = 64
# Structure max token is padding (4099). We use 4100.
CUT_TOKEN_ID_STRUCT = 4100
# SASA uses bins 0-15 and 0 as mask/pad. We use 32 safely.
CUT_TOKEN_ID_SASA = 32

class MimirTokenizer:
    def __init__(self, tokenizer_collection):
        self.sequence = tokenizer_collection.sequence
        self.structure = tokenizer_collection.structure
        self.sasa = tokenizer_collection.sasa
        
        # New cut tokens
        self.cut_seq = CUT_TOKEN_ID_SEQ
        self.cut_struct = CUT_TOKEN_ID_STRUCT
        self.cut_sasa = CUT_TOKEN_ID_SASA
        
        # Base special tokens
        self.seq_bos = self.sequence.bos_token_id
        self.seq_eos = self.sequence.eos_token_id
        self.seq_pad = self.sequence.pad_token_id
        self.seq_mask = self.sequence.mask_token_id
        
        self.struct_pad = self.structure.pad_token_id
        self.struct_mask = self.structure.mask_token_id
        self.struct_bos = self.structure.bos_token_id
        self.struct_eos = self.structure.eos_token_id
        self.struct_nan = 2246 # Explicitly defined NaN token for VQ-VAE structure space
        
        # Note: In ESM3, the SASA tokenizer doesn't have distinct special tokens.
        # pad, mask, bos, and eos all genuinely resolve to 0.
        self.sasa_pad = self.sasa.pad_token_id
        self.sasa_mask = self.sasa.mask_token_id
        self.sasa_bos = self.sasa.bos_token_id
        self.sasa_eos = self.sasa.eos_token_id



def load_tokenizer() -> MimirTokenizer:
    """
    Loads the ESM3 base tokenizer and registers the <cut> token on all three tracks.
    Returns the extended tokenizer. Called once at startup.
    Token IDs must be identical across both contexts.
    """
    tokenizers = get_esm3_model_tokenizers("esm3_sm_open_v1")
    return MimirTokenizer(tokenizers)


def build_input_tensors(
    fingerprint: Dict[str, Any],
    binder: Optional[Dict[str, Any]],
    tokenizer: MimirTokenizer,
    binder_len: int = 96,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs the full input tensors for one example.
    
    Args:
        fingerprint: Dict with 'sequence', 'structure_tokens', 'sasa', 'position_ids'
        binder: Dict with 'sequence', and optionally 'structure_tokens' and 'sasa', 
                or None for inference mode.
        tokenizer: MimirTokenizer instance.
        binder_len: Length of the generated binder. Used in inference mode when binder is None.
    
    Returns:
        (seq_tokens, struct_tokens, sasa_tokens, position_ids, attention_mask)
    """
    # 1. Fingerprint Processing
    # Sequence mapping
    fp_seq_tokens = tokenizer.sequence.encode(fingerprint["sequence"])
    # encode() adds BOS (0) and EOS (2), we strip them
    if len(fp_seq_tokens) >= 2 and fp_seq_tokens[0] == tokenizer.seq_bos and fp_seq_tokens[-1] == tokenizer.seq_eos:
        fp_seq_tokens = fp_seq_tokens[1:-1]
        
    fp_seq_tensor = torch.tensor(fp_seq_tokens, dtype=torch.long)
    fp_struct_tensor = torch.tensor(fingerprint["structure_tokens"], dtype=torch.long)
    fp_sasa_list = [float(x) for x in fingerprint["sasa"]]
    sasa_encoded = tokenizer.sasa.encode(fp_sasa_list, add_special_tokens=False)
    if isinstance(sasa_encoded, torch.Tensor):
        fp_sasa_tensor = sasa_encoded.clone().detach().to(torch.long)
    else:
        fp_sasa_tensor = torch.tensor(sasa_encoded, dtype=torch.long)

    # Position IDs mapping
    fp_pos_ids = fingerprint["position_ids"]
    
    fp_len = len(fp_pos_ids)
    
    # 2. Binder Processing
    if binder is not None:
        binder_seq = binder["sequence"]
        binder_len = len(binder_seq)
        
        binder_seq_tokens = tokenizer.sequence.encode(binder_seq)
        if len(binder_seq_tokens) >= 2 and binder_seq_tokens[0] == tokenizer.seq_bos and binder_seq_tokens[-1] == tokenizer.seq_eos:
            binder_seq_tokens = binder_seq_tokens[1:-1]
            
        binder_seq_tensor = torch.tensor(binder_seq_tokens, dtype=torch.long)
        
        if binder.get("structure_tokens") is not None:
            # Binder has structure. 
            # "SASA is withheld on the binder side even if computable" -> Masked
            binder_struct_tensor = torch.tensor(binder["structure_tokens"], dtype=torch.long)
            binder_sasa_tensor = torch.full((binder_len,), tokenizer.sasa_mask, dtype=torch.long)
        else:
            # Binder without structure
            binder_struct_tensor = torch.full((binder_len,), tokenizer.struct_mask, dtype=torch.long)
            binder_sasa_tensor = torch.full((binder_len,), tokenizer.sasa_mask, dtype=torch.long)
    else:
        # Inference mode: Binder is None. We use the passed binder_len
        # to generate fully masked binder tracks.
        binder_seq_tensor = torch.full((binder_len,), tokenizer.seq_mask, dtype=torch.long)
        binder_struct_tensor = torch.full((binder_len,), tokenizer.struct_mask, dtype=torch.long)
        binder_sasa_tensor = torch.full((binder_len,), tokenizer.sasa_mask, dtype=torch.long)

    # 3. Concatenation
    # [BOS] + [protein fingerprint] + [CUT] + [binder] + [EOS]
    
    # Sequence track
    seq_track = torch.cat([
        torch.tensor([tokenizer.seq_bos], dtype=torch.long),
        fp_seq_tensor,
        torch.tensor([tokenizer.cut_seq], dtype=torch.long),
        binder_seq_tensor,
        torch.tensor([tokenizer.seq_eos], dtype=torch.long)
    ])
    
    # Structure track
    struct_track = torch.cat([
        torch.tensor([tokenizer.struct_bos], dtype=torch.long),
        fp_struct_tensor,
        torch.tensor([tokenizer.cut_struct], dtype=torch.long),
        binder_struct_tensor,
        torch.tensor([tokenizer.struct_eos], dtype=torch.long)
    ])
    
    # SASA track
    sasa_track = torch.cat([
        torch.tensor([tokenizer.sasa_bos], dtype=torch.long),
        fp_sasa_tensor,
        torch.tensor([tokenizer.cut_sasa], dtype=torch.long),
        binder_sasa_tensor,
        torch.tensor([tokenizer.sasa_eos], dtype=torch.long)
    ])
    
    # Position IDs
    # Fingerprint position IDs + 1000 for CUT + 1001, 1002... for Binder
    last_fp_pos = fp_pos_ids[-1] if fp_len > 0 else 0
    cut_pos = last_fp_pos + 1000
    binder_positions = list(range(cut_pos + 1, cut_pos + 1 + binder_len))
    eos_pos = binder_positions[-1] + 1 if binder_len > 0 else cut_pos + 1
    
    pos_ids = [0] + fp_pos_ids + [cut_pos] + binder_positions + [eos_pos]
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long)
    
    # Attention mask (1s for all real tokens before arbitrary batch padding)
    attention_mask = torch.ones(len(seq_track), dtype=torch.long)
    
    return seq_track, struct_track, sasa_track, pos_ids_tensor, attention_mask

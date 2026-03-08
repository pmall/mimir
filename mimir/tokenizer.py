"""
Tokenizer and dataloader utilities for Mimir v2.
"""

from typing import Dict, Any, Tuple, Optional
import torch
from esm.tokenization import get_esm3_model_tokenizers

class MimirTokenizer:
    def __init__(self, tokenizer_collection):
        self.sequence = tokenizer_collection.sequence
        self.structure = tokenizer_collection.structure
        self.sasa = tokenizer_collection.sasa
        
        # Sequence track
        self.seq_bos = self.sequence.bos_token_id  # 0
        self.seq_eos = self.sequence.eos_token_id  # 2
        self.seq_pad = self.sequence.pad_token_id  # 1
        self.seq_mask = self.sequence.mask_token_id  # 32
        self.seq_chainbreak = self.sequence.chain_break_token_id  # 31
        
        # Structure track
        self.struct_pad = self.structure.pad_token_id  # 4099
        self.struct_mask = self.structure.mask_token_id  # 4096
        self.struct_bos = self.structure.bos_token_id  # 4098
        self.struct_eos = self.structure.eos_token_id  # 4097
        self.struct_chainbreak = self.structure.chain_break_token_id  # 4100
        self.struct_nan = 2246  # NaN token for VQ-VAE structure space
        
        # SASA track
        # Note: In ESM3, the SASA tokenizer doesn't have distinct special tokens.
        # pad, mask, bos, and eos all genuinely resolve to 0.
        self.sasa_pad = self.sasa.pad_token_id  # 0
        self.sasa_mask = self.sasa.mask_token_id  # 0
        self.sasa_bos = self.sasa.bos_token_id  # 0
        self.sasa_eos = self.sasa.eos_token_id  # 0
        self.sasa_chainbreak = self.sasa.pad_token_id  # 0



def load_tokenizer() -> MimirTokenizer:
    """
    Loads the ESM3 base tokenizer with chainbreak tokens.
    Returns the tokenizer. Called once at startup.
    """
    tokenizers = get_esm3_model_tokenizers("esm3_sm_open_v1")
    return MimirTokenizer(tokenizers)


def build_input_tensors(
    fingerprint: Dict[str, Any],
    binder: Optional[Dict[str, Any]],
    tokenizer: MimirTokenizer,
    binder_len: int = 96,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs the full input tensors for one example.
    
    Args:
        fingerprint: Dict with 'sequence', 'structure_tokens', 'sasa', 'coordinates'
        binder: Dict with 'sequence', and optionally 'structure_tokens' and 'sasa', 
                or None for inference mode.
        tokenizer: MimirTokenizer instance.
        binder_len: Length of the generated binder. Used in inference mode when binder is None.
    
    Returns:
        (seq_tokens, struct_tokens, sasa_tokens, attention_mask, chain_id, structure_coords)
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

    # Fingerprint length
    fp_len = len(fp_seq_tensor)
    
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
    # [BOS] + [protein fingerprint] + [chainbreak] + [binder] + [EOS]
    
    # Sequence track
    seq_track = torch.cat([
        torch.tensor([tokenizer.seq_bos], dtype=torch.long),
        fp_seq_tensor,
        torch.tensor([tokenizer.seq_chainbreak], dtype=torch.long),
        binder_seq_tensor,
        torch.tensor([tokenizer.seq_eos], dtype=torch.long)
    ])
    
    # Structure track
    struct_track = torch.cat([
        torch.tensor([tokenizer.struct_bos], dtype=torch.long),
        fp_struct_tensor,
        torch.tensor([tokenizer.struct_chainbreak], dtype=torch.long),
        binder_struct_tensor,
        torch.tensor([tokenizer.struct_eos], dtype=torch.long)
    ])
    
    # SASA track
    sasa_track = torch.cat([
        torch.tensor([tokenizer.sasa_bos], dtype=torch.long),
        fp_sasa_tensor,
        torch.tensor([tokenizer.sasa_chainbreak], dtype=torch.long),
        binder_sasa_tensor,
        torch.tensor([tokenizer.sasa_eos], dtype=torch.long)
    ])
    
    # Chain ID: 1 for fingerprint + chainbreak, 2 for binder + EOS
    chain_id = torch.cat([
        torch.ones(1 + fp_len + 1, dtype=torch.long),  # BOS + fingerprint + chainbreak
        torch.full((binder_len + 1,), 2, dtype=torch.long)  # binder + EOS
    ])
    
    # Structure coordinates: shape (L, 3, 3)
    # NaN for BOS, chainbreak, binder, EOS; real coords for fingerprint
    L = len(seq_track)
    fp_coords = torch.tensor(fingerprint["coordinates"], dtype=torch.float32)  # (fp_len, 3, 3)
    
    structure_coords = torch.full((L, 3, 3), float('nan'), dtype=torch.float32)
    # Fill in fingerprint coordinates at positions 1 to fp_len
    structure_coords[1:1 + fp_len] = fp_coords
    
    # Attention mask (1s for all real tokens before arbitrary batch padding)
    attention_mask = torch.ones(len(seq_track), dtype=torch.long)
    
    return seq_track, struct_track, sasa_track, attention_mask, chain_id, structure_coords

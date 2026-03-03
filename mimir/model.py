"""
Model definitions and loading utilities for Mimir v2.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from peft import get_peft_model, LoraConfig

from mimir.tokenizer import CUT_TOKEN_ID_SEQ, CUT_TOKEN_ID_STRUCT, CUT_TOKEN_ID_SASA

logger = logging.getLogger(__name__)


class ExtendedEmbedding(nn.Module):
    """
    Wraps an existing ESM3 embedding layer to seamlessly inject a novel <cut> token
    embedding, while keeping the original layer's weights frozen.
    """
    def __init__(self, original_embedding: nn.Embedding, cut_token_id: int):
        super().__init__()
        self.original_embedding = original_embedding
        self.cut_token_id = cut_token_id
        
        # Ensure the original embedding is completely frozen
        self.original_embedding.weight.requires_grad = False
        
        # Create a new, trainable embedding solely for our <cut> token
        embed_dim = original_embedding.embedding_dim
        self.cut_embedding = nn.Embedding(1, embed_dim)
        
        # Initialize randomly (could also initialize to pad token or similar)
        nn.init.normal_(self.cut_embedding.weight, mean=0, std=embed_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routes the indices logically:
        1. Replace <cut> tokens with 0 in the input stream before querying the frozen embedding.
        2. Query the original frozen embedding.
        3. Replace the embeddings at the <cut> token positions with our new embedding.
        """
        # Create a boolean mask of where the cut token is (shape: batch, seq_len)
        cut_mask = (x == self.cut_token_id)
        
        # To avoid out-of-bounds queries on the frozen embedding, we temporarily
        # replace the cut_token_id with 0 (which is safe).
        safe_x = x.clone()
        safe_x[cut_mask] = 0
        
        # Query the original, frozen embedding (gradients will not pass through here)
        base_embeds = self.original_embedding(safe_x)
        
        result = base_embeds.clone()
        if cut_mask.any():
            result[cut_mask] = self.cut_embedding.weight[0]
            
        return result


def load_model(checkpoint_path: Optional[str] = None) -> nn.Module:
    """
    Loads ESM3 1.4B, attaches LoRA adapters via PEFT (r=16, alpha=32, dropout=0.1), 
    registers the <cut> token embeddings on all three tracks, and optionally 
    restores weights from a checkpoint. Returns the model ready for forward passes.
    
    Args:
        checkpoint_path: Optional path to a directory containing PEFT weights and
                         extended embedding weights.
                         
    Returns:
        The configured model.
    """
    # 1. Load the frozen base model
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model = model.bfloat16()
    
    # Freeze all base parameters explicitly
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Inject the custom <cut> token embeddings into the encoder
    model.encoder.sequence_embed = ExtendedEmbedding(
        model.encoder.sequence_embed, cut_token_id=CUT_TOKEN_ID_SEQ
    )
    model.encoder.structure_tokens_embed = ExtendedEmbedding(
        model.encoder.structure_tokens_embed, cut_token_id=CUT_TOKEN_ID_STRUCT
    )
    model.encoder.sasa_embed = ExtendedEmbedding(
        model.encoder.sasa_embed, cut_token_id=CUT_TOKEN_ID_SASA
    )
    
    # 3. Apply PEFT LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["layernorm_qkv.1", "out_proj", "proj", "ffn.1", "ffn.3"], # targeting all linear layers in ESM3 transformer block
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    )
    
    # This automatically adds the adapters and sets requires_grad=True only for them
    model = get_peft_model(model, peft_config)
    
    # Because we hijacked the embeddings, we need to explicitly ensure their parameters
    # still require gradients after PEFT initialization (PEFT might freeze non-adapter parts)
    model.base_model.model.encoder.sequence_embed.cut_embedding.weight.requires_grad = True
    model.base_model.model.encoder.structure_tokens_embed.cut_embedding.weight.requires_grad = True
    model.base_model.model.encoder.sasa_embed.cut_embedding.weight.requires_grad = True
    
    # print trainable parameters summary if logger is at info or below
    if logger.getEffectiveLevel() <= logging.INFO:
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # 4. Resume from checkpoint if provided
    if checkpoint_path is not None:
        model.from_pretrained(checkpoint_path)
        
        cut_embedding_keys = [
            "base_model.model.encoder.sequence_embed.cut_embedding.weight",
            "base_model.model.encoder.structure_tokens_embed.cut_embedding.weight",
            "base_model.model.encoder.sasa_embed.cut_embedding.weight",
        ]
        cut_embeddings = torch.load(os.path.join(checkpoint_path, "cut_embeddings.pt"), map_location="cpu")
        
        missing_keys, unexpected_keys = model.load_state_dict(cut_embeddings, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
    return model

"""
Model definitions and loading utilities for Mimir v2.
"""

import logging
from typing import Optional

import torch.nn as nn
from esm.models.esm3 import ESM3
from peft import get_peft_model, LoraConfig

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Optional[str] = None) -> nn.Module:
    """
    Loads ESM3 1.4B, attaches LoRA adapters via PEFT (r=16, alpha=32, dropout=0.1),
    and optionally restores weights from a checkpoint. Returns the model ready for forward passes.
    
    Args:
        checkpoint_path: Optional path to a directory containing PEFT weights.
                        
    Returns:
        The configured model.
    """
    # 1. Load the base model
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model = model.bfloat16()
    
    # Freeze all base parameters explicitly
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Apply PEFT LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["layernorm_qkv.1", "out_proj", "proj", "ffn.1", "ffn.3"],
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    )
    
    # This automatically adds the adapters and sets requires_grad=True only for them
    model = get_peft_model(model, peft_config)
    
    # print trainable parameters summary if logger is at info or below
    if logger.getEffectiveLevel() <= logging.INFO:
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # 3. Resume from checkpoint if provided
    if checkpoint_path is not None:
        model.from_pretrained(checkpoint_path)
        
    return model

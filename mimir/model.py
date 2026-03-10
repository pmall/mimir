"""
Model definitions and loading utilities for Mimir v2.
"""

import logging
from typing import Optional

import torch.nn as nn
from esm.models.esm3 import ESM3
from peft import PeftModel, get_peft_model, LoraConfig

logger = logging.getLogger(__name__)


# --- Constants ---

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["layernorm_qkv.1", "out_proj", "proj", "ffn.1", "ffn.3"],
    lora_dropout=0.1,
    bias="none",
    task_type=None,
)


def load_model(checkpoint_path: Optional[str] = None) -> nn.Module:
    """Loads ESM3 1.4B, attaches LoRA adapters via PEFT (r=16, alpha=32, dropout=0.1),
    and optionally restores weights from a checkpoint. Returns the model ready for forward passes.

    Args:
        checkpoint_path: Optional path to a directory containing PEFT adapter weights
                        (adapter_config.json + adapter_model.safetensors).

    Returns:
        The configured model.
    """
    # 1. Load the base model
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model = model.bfloat16()

    # Freeze all base parameters explicitly
    for param in model.parameters():
        param.requires_grad = False

    # 2. Apply PEFT LoRA — either fresh or from a checkpoint
    if checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        logger.info(f"Loaded adapter weights from {checkpoint_path}")
    else:
        model = get_peft_model(model, LORA_CONFIG)

    # Log trainable parameters summary
    if logger.getEffectiveLevel() <= logging.INFO:
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable:,d} / {total:,d} "
            f"({100 * trainable / total:.2f}%)"
        )

    # Enable gradient checkpointing to save VRAM (may not be available on all backends)
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        logger.warning("gradient_checkpointing_enable not available, skipping")

    return model

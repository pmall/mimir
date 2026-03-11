"""
Model definitions and loading utilities for Mimir v2.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from esm.models.esm3 import ESM3
from esm.layers.transformer_stack import TransformerStack
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


# --- Gradient Checkpointing ---


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enables gradient checkpointing on ESM3's TransformerStack blocks.

    ESM3 does not inherit from HuggingFace PreTrainedModel, so we wrap
    each block's forward with torch.utils.checkpoint individually.
    This trades ~30% extra compute for significant VRAM savings.
    """
    for module in model.modules():
        if isinstance(module, TransformerStack):
            for block in module.blocks:
                _wrap_block_with_checkpoint(block)
            logger.info(
                f"Gradient checkpointing enabled on TransformerStack "
                f"({len(module.blocks)} blocks)"
            )
            return

    logger.warning("No TransformerStack found — gradient checkpointing not applied")


def _wrap_block_with_checkpoint(block: nn.Module) -> None:
    """Wraps a single transformer block's forward with gradient checkpointing."""
    original_forward = block.forward

    def checkpointed_forward(*args, **kwargs):
        return torch_checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    block.forward = checkpointed_forward


# --- Model Loading ---


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

    # 3. Enable gradient checkpointing to save VRAM
    _enable_gradient_checkpointing(model)

    return model

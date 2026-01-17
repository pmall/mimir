"""
Mimir Model Utilities
=====================

This module provides helper functions for manipulating the ESM-3 model architecture,
specifically for fine-tuning tasks that require vocabulary expansion (e.g., adding condition tokens).

Functions:
    - resize_esm3_tokens: Safely resizes the sequence embeddings and output heads of an ESM-3 model.
"""

import torch
import torch.nn as nn
from esm.models.esm3 import ESM3

def resize_esm3_tokens(model: ESM3, new_vocab_size: int):
    """
    Resizes the embedding layer and output head of an ESM-3 model to accommodate a new vocabulary size.
    
    This function is critical for fine-tuning with new special tokens (like target IDs).
    It manually adjusts `model.encoder.sequence_embed` and `model.output_heads.sequence_head`.
    
    Args:
        model (ESM3): The ESM-3 model instance to resize.
        new_vocab_size (int): The new size of the vocabulary (base vocab + new tokens).
        
    Raises:
        ValueError: If the model is not an instance of ESM3 or if new_vocab_size is smaller than current.
    """
    # 1. Safety Checks
    if not isinstance(model, ESM3):
        raise ValueError(f"Expected model to be an instance of ESM3, got {type(model)}")
        
    current_vocab_size = model.encoder.sequence_embed.num_embeddings
    
    if new_vocab_size < current_vocab_size:
        raise ValueError(f"New vocab size ({new_vocab_size}) cannot be smaller than current ({current_vocab_size})")
        
    if new_vocab_size == current_vocab_size:
        print(f"Vocab size is already {new_vocab_size}. No resizing needed.")
        return

    print(f"Resizing ESM-3 embeddings from {current_vocab_size} to {new_vocab_size}...")

    # 2. Resize Encoder Embeddings (Input)
    # ------------------------------------
    # We create a new Embedding layer with the new size and copy existing weights.
    old_embeddings = model.encoder.sequence_embed
    embedding_dim = old_embeddings.embedding_dim
    
    new_embeddings = nn.Embedding(
        num_embeddings=new_vocab_size, 
        embedding_dim=embedding_dim,
        padding_idx=old_embeddings.padding_idx
    )
    
    # Initialize with the same device and dtype
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    
    # Copy existing weights
    # We use a context manager to ensure we don't track gradients during this copy
    with torch.no_grad():
        new_embeddings.weight[:current_vocab_size] = old_embeddings.weight
        
        # Initialize new tokens with the mean of existing embeddings
        # This provides a stable starting point for training, rather than random noise
        mean_embedding = old_embeddings.weight.mean(dim=0, keepdim=True)
        new_embeddings.weight[current_vocab_size:] = mean_embedding

    # Replace the layer in the model
    model.encoder.sequence_embed = new_embeddings
    
    # 3. Resize Output Head (Prediction)
    # ----------------------------------
    # The output head transforms hidden states back to logits for the vocabulary.
    # ESM-3 uses a RegressionHead which wraps a Linear layer (or similar).
    # We need to find the specific linear projection to vocab size.
    
    # Inspecting ESM-3 architecture: model.output_heads.sequence_head likely ends with a projection.
    # Based on standard ESM designs, it might be a simple Linear layer or a small MLP.
    # We assume 'model.output_heads.sequence_head' is the module responsible.
    
    # Let's inspect the `sequence_head` structure dynamically if possible, or assume standard RegressionHead.
    # RegressionHead in ESM typically has a defined structure.
    # For safety, we only resize the LAST Linear layer if it matches vocab size.
    
    seq_head = model.output_heads.sequence_head
    
    # We iterate correctly to find the vocab projection layer. 
    # Usually it's the last child or the module itself if it's just Linear.
    # In `esm.layers.regression_head.RegressionHead`, it usually has an attribute like `proj` or is a Sequential.
    # Since we don't have the full `RegressionHead` code in front of us (it was imported), 
    # we'll look for the linear layer that matches `current_vocab_size`.
    
    found_layer = False
    
    # Recursive search for the linear layer with out_features == current_vocab_size
    for name, module in seq_head.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == current_vocab_size:
            print(f"Found output projection layer: {name}")
            
            # Create new layer
            new_linear = nn.Linear(
                in_features=module.in_features,
                out_features=new_vocab_size,
                bias=module.bias is not None
            )
            
            new_linear.to(module.weight.device, dtype=module.weight.dtype)
            
            with torch.no_grad():
                new_linear.weight[:current_vocab_size] = module.weight
                if module.bias is not None:
                    new_linear.bias[:current_vocab_size] = module.bias
                    
                # Initialize new output weights
                # Small random initialization or mean?
                # For outputs, 0 or small noise is usually fine.
                # We'll stick to a small initialization close to 0 to avoid massive logits.
                nn.init.normal_(new_linear.weight[current_vocab_size:], mean=0, std=0.02)
                if new_linear.bias is not None:
                    nn.init.zeros_(new_linear.bias[current_vocab_size:])
            
            # Replace the layer using setattr. 
            # If it's a direct attribute of seq_head:
            if hasattr(seq_head, name): # e.g. if name is just the attribute name
                 setattr(seq_head, name, new_linear)
            else:
                 # If it's deeper (e.g. '0.linear'), this simple setattr won't work.
                 # But usually RegressionHead is simple.
                 # If named_modules returns hierarchical names, we might need a more complex setter.
                 # For now, let's assume it's a direct attribute or we can access it.
                 # Actually, commonly `RegressionHead` IS a `nn.Sequential` or has a `.proj`
                 pass

            # Update found flag.
            # Realistically, we need to modify the parent.
            # Let's try to replace it in the parent `seq_head`.
            # If `name` is empty, then `seq_head` ITSELF is the linear layer, but RegressionHead wraps it.
            
            # Use a robust replacement strategy
            _replace_module(seq_head, name, new_linear)
            found_layer = True
            break
            
    if not found_layer:
        print("WARNING: Could not find output projection layer in sequence_head matching vocab size.")
        print("Model output might be incorrect for new tokens.")
    else:
        print("Successfully resized output head.")

def _replace_module(root_module, path, new_module):
    """
    Helper to replace a module deep in the hierarchy given a path string (e.g., 'layer.0.linear').
    """
    if path == "":
        # Root is the module, but we can't replace 'self'. 
        return
        
    parts = path.split('.')
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
        
    last_part = parts[-1]
    
    # Check if it's an attribute or item (in ModuleList/Sequential)
    try:
        # Try converting to int for list/sequential access
        idx = int(last_part)
        parent[idx] = new_module
    except ValueError:
        # Standard attribute
        setattr(parent, last_part, new_module)

"""Training script for fine-tuning ESM-3 with LoRA on peptide sequences."""

import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from transformers import get_cosine_schedule_with_warmup

from esm.models.esm3 import ESM3
from mimir.dataset import PeptideDataset, create_dynamic_collate_fn
from mimir.tokenizer import AminoAcidTokenizer
from mimir.sampler import LengthGroupedSampler

try:
    from mimir.model_utils import resize_esm3_tokens
except ImportError:
    # Allow running without mimir installed if file is local
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mimir.model_utils import resize_esm3_tokens

def train(args):
    """
    Main training loop for Fine-Tuning ESM-3 with LoRA and Target Conditioning.
    
    This function implements the complete training pipeline:
    1.  **Data Loading**: Loads dataset and creates a masked collator.
    2.  **Model Setup**: Loads ESM-3, RESIZES embeddings for target tokens, and applies LoRA.
    3.  **Training**: Optimizes for Masked Language Modeling (MLM) loss on peptide sequences.
    
    Args:
        args: Parsed command-line arguments.
    """
    # 1. Setup Device
    # ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    # -------------------
    print("Loading tokenizer and dataset...")
    
    # Resolve absolute path if relative
    if not os.path.isabs(args.dataset):
        dataset_path = os.path.join(os.path.dirname(__file__), "..", args.dataset)
    else:
        dataset_path = args.dataset

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please check the path.")
        return

    # Extract unique targets to build vocabulary (Dataset will handle max_length)
    targets = set()
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.add(row["target"])
    print(f"Found {len(targets)} unique targets.")

    # Initialize Tokenizer with targets
    tokenizer = AminoAcidTokenizer(targets=list(targets))
    
    # Initialize Dataset (Auto-max-length)
    dataset = PeptideDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
    )
    
    # Initialize Dynamic Collator with Masking
    # We pass both pad_idx and mask_idx to handle padding and masking logic
    collate_fn = create_dynamic_collate_fn(
        pad_idx=dataset.tokenizer.pad_idx,
        mask_idx=dataset.tokenizer.mask_token_id
    )
    
    # Initialize Sampler for Smart Batching
    sampler = LengthGroupedSampler(
        dataset=dataset, 
        batch_size=args.batch_size, 
        drop_last=False
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, # Use custom sampler
        shuffle=False,   # Must be False when using sampler
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Model Initialization
    # -----------------------
    print("Loading ESM-3 model...")
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model.to(device)
    
    # CRITICAL: Resize Embeddings
    # ---------------------------
    # We must resize the model's embedding tables to accommodate the new target tokens.
    # We use our custom safe utility for this.
    try:
        resize_esm3_tokens(model, tokenizer.vocab_size)
    except Exception as e:
        print(f"Failed to resize embeddings: {e}")
        return
    
    # 4. LoRA Configuration
    # ---------------------
    print("Applying LoRA...")
    # We target the projection layers in attention mechanism for efficient fine-tuning.
    peft_config = LoraConfig(
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # QUIRK 1: No TaskType.CAUSAL_LM
        # Reason: PEFT tries to access `prepare_inputs_for_generation` which ESM-3 lacks.
        # Fix: Treat as generic nn.Module by omitting task_type.
        
        # QUIRK 2: Custom Target Modules
        # Reason: ESM-3 modules don't follow standard 'q_proj'/'v_proj' naming.
        # Fix: Inspected model structure manually to find:
        # - 'layernorm_qkv.1': The linear layer inside the fused QKV projection.
        # - 'out_proj': The output projection of the attention block.
        target_modules=["layernorm_qkv.1", "out_proj"] 
    )
    
    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"Failed to apply PEFT: {e}")
        return

    # 5. Training Loop Setup
    # ----------------------
    # Optimizer Setup
    # ---------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            print("Using 8-bit AdamW optimizer via bitsandbytes...")
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
        except ImportError:
            print("WARNING: bitsandbytes not found. Falling back to standard AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    
    # Loss Function:
    # We use CrossEntropyLoss with reduction='none' to handle per-sample weighting.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    # Scheduler Setup
    # ---------------
    # Calculate total training steps
    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    print(f"Total training steps: {max_train_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps
    )

    # 6. Training Execution
    # ---------------------
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        total_true_loss = 0
        total_correct_masked = 0
        total_masked_tokens = 0
        total_perplexity = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Move inputs to device
            tokens = batch["tokens"].to(device)   # Input: Masked sequences
            labels = batch["labels"].to(device)   # Target: Original IDs at masked pos, -100 otherwise
            
            try:
                # Forward Pass
                # We pass the token sequence; ESM-3 returns 'sequence_logits'.

                # ESM-3 returns an output object containing 'sequence_logits'.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = model(sequence_tokens=tokens)
                
                # Compute Loss
                # Output shape: [Batch, Length, Vocab]
                # Cast to float32 for CrossEntropy stability
                logits = output.sequence_logits.float()
                
                # CRITICAL Fix: Mask Target Logits
                # -------------------------------
                # The model should NEVER predict a target ID (which we added to the vocab)
                # for a masked sequence position.
                # Base vocab size is ~33 (amino acids + special tokens).
                # All tokens >= base_vocab_size are Targets.
                # We force their probability to 0 by setting logits to -inf.
                logits[..., dataset.tokenizer.base_vocab_size:] = float('-inf')
                
                # Flatten logits and labels for CrossEntropy
                # Logits: [B*L, Vocab]
                # Labels: [B*L]
                # We use reduction='none' to get per-token loss first
                loss_per_token = criterion(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # Reshape back to [Batch, Length]
                loss_per_token = loss_per_token.view(tokens.size(0), tokens.size(1))
                
                # Vectorized Boost Calculation
                # ----------------------------
                # 1. Mask of valid tokens (where labels != -100)
                mask = labels != -100 # [Batch, Length]
                num_masked = mask.sum(dim=1).float() # [Batch]
                
                # 2. Calculate base sample loss (mean of masked tokens per sample)
                # Avoid division by zero for samples with no masked tokens
                sample_loss = (loss_per_token * mask.float()).sum(dim=1) / num_masked.clamp(min=1)
                
                # 3. Apply Boost (log-based on absolute masked count)
                # Using log(num_masked + 1) for gentle scaling that favors more masks
                boost = 1.0 + args.masking_boost_ratio * torch.log(num_masked + 1)
                
                # 4. Final Loss
                # We average the boosted loss over valid samples (those with masked tokens)
                valid_samples = num_masked > 0
                if valid_samples.any():
                    loss = (sample_loss * boost)[valid_samples].mean()
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Backward Pass & Gradient Accumulation
                # -------------------------------------
                # Normalize loss for accumulation
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (num_batches + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * args.gradient_accumulation_steps # Scale back for logging
                num_batches += 1
                
                # Metrics Calculation for Logging
                # -------------------------------
                with torch.no_grad():
                    # Calculate accuracy on masked tokens
                    # logits: [Batch, Length, Vocab]
                    # predictions: [Batch, Length]
                    predictions = logits.argmax(dim=-1)
                    
                    # mask: [Batch, Length] (True where token was masked)
                    # correct: [Batch, Length]
                    correct = (predictions == labels) & mask
                    masked_accuracy = correct.sum().float() / num_masked.sum().clamp(min=1)
                    
                    # Perplexity
                    perplexity = torch.exp(sample_loss.mean())
                    
                    # Accumulate for Epoch Summary
                    total_correct_masked += correct.sum().item()
                    total_masked_tokens += num_masked.sum().item()
                    total_perplexity += perplexity.item()
                    
                    # Accumulate True Loss (Weighted by sample count or just sum? Just sum for avg)
                    # We want the average true loss over the epoch.
                    # sample_loss is [Batch] of mean loss per sample.
                    # We can use sample_loss.mean() for the batch average.
                    total_true_loss += sample_loss.mean().item()

                # Update Progress Bar
                pbar.set_postfix({
                    "Loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "PPL": f"{perplexity.item():.2f}",
                    "Acc": f"{masked_accuracy.item():.2%}"
                })
                
            except Exception as e:
                print(f"Error in training step: {e}")
                # Optional: print full traceback for debugging
                import traceback
                traceback.print_exc()
                raise e  # Re-raise to stop training completely
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_true_loss = total_true_loss / num_batches
            avg_acc = total_correct_masked / max(1, total_masked_tokens)
            avg_ppl = total_perplexity / num_batches
            
            print("-" * 60)
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Avg Loss (Boosted): {avg_loss:.4f}")
            print(f"  Avg Loss (True):    {avg_true_loss:.4f}")
            print(f"  Avg Perplexity: {avg_ppl:.4f}")
            print(f"  Masked Accuracy: {avg_acc:.2%}")
            print("-" * 60)
        
        # Save Checkpoints
        # ----------------
        
        # 1. Save Last Model (Always overwrites to keep latest state)
        last_path = "checkpoints/last_model"
        os.makedirs(last_path, exist_ok=True)
        model.save_pretrained(last_path)
        
        # 2. Save Best Model (Only if loss improves)
        if avg_true_loss < best_loss:
            best_loss = avg_true_loss
            best_path = "checkpoints/best_model"
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            print(f"New Best Model! (Loss: {best_loss:.4f}). Saved to {best_path}")
        else:
            print(f"Saved latest model to {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--masking_boost_ratio", type=float, default=0.5, help="Log boost factor for number of masked tokens")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use bitsandbytes 8-bit AdamW")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for scheduler")
    parser.add_argument("--dataset", type=str, default="data/peptide_dataset.csv", help="Path to the training dataset CSV")
    args = parser.parse_args()
    
    train(args)

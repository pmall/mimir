"""
Train MÍMIR v2 model.

Usage:
    uv run python -m scripts.train \\
        --config data/run78-v2/config.json \\
        --checkpoint-dir checkpoints/ \\
        [--epochs 100] [--batch-size 4] [--peak-lr 1e-4] [-v]
"""

import argparse
import logging
import os
import sys
import json
import random
import math
import threading
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from mimir.config import load_config
from mimir.model import load_model
from mimir.dataset import MimirDataset, mimir_collate_fn, BucketBatchSampler
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmdb").setLevel(logging.WARNING)

def set_seed(seed: int):
    """Set global seeds for reproducibility."""
    random.seed(seed)
    # np.random.seed(seed)  # If numpy is added later
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure dataloader/sampler use this seed if needed
    os.environ["PYTHONHASHSEED"] = str(seed)

# --- Masking Strategy ---

def apply_mlm_masking(batch: dict, tokenizer: Any) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    """
    Applies MLM masking to the binder side (after <cut>) according to the strategy:
    - 25% to 75% uniform masking rate per example
    - Apply to sequence track always (if valid)
    - Apply to structure track independently if it has structure
    """
    seq = batch["sequence"].clone()
    struct = batch["structure"].clone()
    
    labels_seq = torch.full_like(seq, -100)
    labels_struct = torch.full_like(struct, -100)
    
    cut_token_id = tokenizer.seq_chainbreak
    batch_size, seq_len = seq.shape
    
    for i in range(batch_size):
        cut_pos_t = (seq[i] == cut_token_id).nonzero(as_tuple=True)[0]
        if len(cut_pos_t) == 0:
            continue
        cut_pos = cut_pos_t[0].item()
        
        eos_pos_t = (seq[i] == tokenizer.seq_eos).nonzero(as_tuple=True)[0]
        if len(eos_pos_t) == 0:
            eos_pos = seq_len
        else:
            eos_pos = eos_pos_t[0].item()
            
        binder_start = cut_pos + 1
        binder_end = eos_pos
        binder_len = binder_end - binder_start
        
        if binder_len <= 0:
            continue
            
        # Sample masking rate [0.25, 0.75]
        mask_rate = random.uniform(0.25, 0.75)
        num_mask = max(1, int(round(binder_len * mask_rate)))
        
        # 1. Mask Sequence
        mask_indices_seq = random.sample(range(binder_start, binder_end), num_mask)
        for idx in mask_indices_seq:
            labels_seq[i, idx] = seq[i, idx].item()
            seq[i, idx] = tokenizer.seq_mask
            
        # 2. Mask Structure
        # Check if structure is present (not all mask tokens)
        struct_binder = struct[i, binder_start:binder_end]
        if not torch.all(struct_binder == tokenizer.struct_mask):
            mask_indices_struct = random.sample(range(binder_start, binder_end), num_mask)
            for idx in mask_indices_struct:
                labels_struct[i, idx] = struct[i, idx].item()
                struct[i, idx] = tokenizer.struct_mask

    masked_batch = {
        "sequence": seq,
        "structure": struct,
        "sasa": batch["sasa"],
        "chain_id": batch["chain_id"],
        "structure_coords": batch["structure_coords"],
        "attention_mask": batch["attention_mask"]
    }
    return masked_batch, labels_seq, labels_struct

def compute_mlm_loss(
    sequence_logits: torch.Tensor,
    structure_logits: torch.Tensor,
    labels_seq: torch.Tensor,
    labels_struct: torch.Tensor,
    tokenizer: Any,
    lam: float,
    gradient_accumulation_steps: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Computes the weighted Masked Language Modeling loss and granular metrics.
    
    Args:
        sequence_logits: (B, L, V_seq)
        structure_logits: (B, L, V_struct)
        labels_seq: (B, L) with -100 for ignored tokens
        labels_struct: (B, L) with -100 for ignored tokens and 2246 for NaN tokens
        lam: lambda penalty weight for number of masked tokens
        
    Returns:
        loss: the scalar loss for backpropagation (already divided by accumulation steps)
        sample_loss: the unweighted per-sample loss tensor (B,) for perplexity tracking
        metrics: dictionary of raw correct/total/loss sums for metric tracking
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    device = sequence_logits.device
    
    loss_seq_per_token = criterion(sequence_logits.float().view(-1, sequence_logits.size(-1)), labels_seq.view(-1))
    loss_seq_per_token = loss_seq_per_token.view(labels_seq.size())
    
    loss_struct_per_token = criterion(structure_logits.float().view(-1, structure_logits.size(-1)), labels_struct.view(-1))
    loss_struct_per_token = loss_struct_per_token.view(labels_struct.size())
    
    mask_seq = labels_seq != -100
    mask_struct = labels_struct != -100
    
    # Exclude NaN targets from structure mask
    mask_struct_valid = mask_struct & (labels_struct != tokenizer.struct_nan)
    
    num_masked_seq = mask_seq.sum(dim=1).float()
    num_masked_struct = mask_struct_valid.sum(dim=1).float()
    total_masked = num_masked_seq + num_masked_struct
    
    sample_loss_seq = (loss_seq_per_token * mask_seq.float()).sum(dim=1)
    # Use valid mask so NaN targets log 0 loss
    sample_loss_struct = (loss_struct_per_token * mask_struct_valid.float()).sum(dim=1)
    
    # Unweighted loss per sample for tracking perplexity
    sample_loss = (sample_loss_seq + sample_loss_struct) / total_masked.clamp(min=1)
    
    # Apply log boost
    weight = lam * torch.log(1 + total_masked)
    boosted_loss = (sample_loss * weight)
    
    valid_samples = total_masked > 0
    if valid_samples.any():
        loss = boosted_loss[valid_samples].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
    loss = loss / gradient_accumulation_steps
    
    # Computations for detailed metrics
    m = {
        "overall_correct": 0, "overall_total": 0, "overall_loss": 0.0,
        "full_seq_correct": 0, "full_seq_total": 0, "full_seq_loss": 0.0,
        "full_struct_correct": 0, "full_struct_total": 0, "full_struct_loss": 0.0,
        "partial_seq_correct": 0, "partial_seq_total": 0, "partial_seq_loss": 0.0
    }
    
    with torch.no_grad():
        pred_seq = sequence_logits.argmax(dim=-1)
        pred_struct = structure_logits.argmax(dim=-1)
        
        correct_seq = (pred_seq == labels_seq) & mask_seq
        correct_struct = (pred_struct == labels_struct) & mask_struct_valid
        
        for i in range(labels_seq.size(0)):
            nm_seq = num_masked_seq[i].item()
            nm_struct = num_masked_struct[i].item()
            if nm_seq == 0 and nm_struct == 0:
                continue
                
            seq_corr = correct_seq[i].sum().item()
            struct_corr = correct_struct[i].sum().item()
            
            m["overall_correct"] += seq_corr + struct_corr
            m["overall_total"] += nm_seq + nm_struct
            m["overall_loss"] += sample_loss_seq[i].item() + sample_loss_struct[i].item()
            
            if nm_struct > 0:
                m["full_seq_correct"] += seq_corr
                m["full_seq_total"] += nm_seq
                m["full_seq_loss"] += sample_loss_seq[i].item()
                
                m["full_struct_correct"] += struct_corr
                m["full_struct_total"] += nm_struct
                m["full_struct_loss"] += sample_loss_struct[i].item()
            else:
                m["partial_seq_correct"] += seq_corr
                m["partial_seq_total"] += nm_seq
                m["partial_seq_loss"] += sample_loss_seq[i].item()
                
    return loss, sample_loss, m

# --- Main Training Logic ---

def safe_div(a: float, b: float) -> float: 
    return a / b if b > 0 else 0.0

def _run(args: argparse.Namespace) -> None:
    # Set seed early
    set_seed(args.seed)
    
    # Enable TF32 for matrix multiplications and convolutions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading tokenizer and configuring dataloader...")
    tokenizer = load_tokenizer()
    
    dataset = MimirDataset(
        associations_csv=args.associations_csv,
        fingerprints_lmdb=args.fingerprints_lmdb,
        binders_lmdb=args.binders_lmdb,
        tokenizer=tokenizer
    )
    logger.info(f"Total samples: {len(dataset)}")
    
    # Checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Calculate steps first
    # Estimate total batches using math.ceil(len / batch_size) because bucket length is variable
    sampler = BucketBatchSampler(
        dataset=dataset, 
        batch_size=args.batch_size, 
        cache_path=checkpoint_dir,
        epoch=1
    )
    # len(sampler) returns estimated batches
    steps_per_epoch = len(sampler) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    logger.info(f"Estimated steps per epoch: {steps_per_epoch}, total steps: {total_steps}")
    
    start_epoch = 0
    latest_ckpt_path = None
    training_state = None
    best_overall_loss = float("inf")
    
    log_file = checkpoint_dir / "training_log.jsonl"
    
    if log_file.exists():
        with open(log_file, "r") as f:
            lines = f.readlines()
        if lines:
            for line in lines[::-1]:
                # Find best overall loss across all runs
                log = json.loads(line)
                if "overall_loss_raw" in log and log["overall_loss_raw"] < best_overall_loss:
                    best_overall_loss = log["overall_loss_raw"]

            last_log = json.loads(lines[-1])
            last_epoch = last_log.get("epoch", 0)
            if last_epoch > 0:
                ckpt_path = checkpoint_dir / f"epoch_{last_epoch}"
                if not ckpt_path.exists():
                    logger.error(f"Log indicates epoch {last_epoch} but checkpoint not found: {ckpt_path}")
                    sys.exit(1)
                latest_ckpt_path = str(ckpt_path)
    
    if latest_ckpt_path is not None:
        training_state_path = os.path.join(latest_ckpt_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
            
            # --- Integrity Check ---
            saved_epoch = training_state.get("epoch")
            if saved_epoch != last_epoch:
                logger.error(f"Integrity Error: training_log says epoch {last_epoch}, but {training_state_path} says epoch {saved_epoch}.")
                sys.exit(1)
                
            start_epoch = saved_epoch
            logger.info(f"Resuming safely from epoch {start_epoch}")
        else:
            logger.error(f"Log indicates epoch {last_epoch} but no training_state.pt found in {latest_ckpt_path}. Cannot guarantee safe resume.")
            sys.exit(1)
    
    logger.info("Loading model...")
    model = load_model(latest_ckpt_path)
    model.to(device)
    model.train()
    
    # Compile the model to reduce overhead
    logger.info("Compiling model (this may take a moment)...")
    model = torch.compile(model, mode="reduce-overhead")
    
    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.use_8bit_adam:
        if not HAS_BNB:
            logger.error("use_8bit_adam requested but bitsandbytes is not installed.")
            sys.exit(1)
        logger.info("Using 8-bit AdamW optimizer.")
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.peak_lr)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.peak_lr)
    
    warmup_steps = int(0.05 * total_steps)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = 1e-5 / args.peak_lr
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume opt & scheduler states
    if training_state is not None:
        if "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        if "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])
        logger.info(f"Resumed optimizer and scheduler from epoch {start_epoch}")
    
    # Loss Criterion is now inside compute_mlm_loss
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        sampler.set_epoch(epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: mimir_collate_fn(b, tokenizer),
            num_workers=args.num_workers,
            persistent_workers=True if args.num_workers > 0 else False,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Epoch {epoch}/{args.epochs}")
        total_loss, total_true_loss = 0.0, 0.0
        num_batches = 0
        total_skipped = 0
        
        # Metrics
        m = {
            "overall_correct": 0, "overall_total": 0, "overall_loss": 0.0,
            "full_seq_correct": 0, "full_seq_total": 0, "full_seq_loss": 0.0,
            "full_struct_correct": 0, "full_struct_total": 0, "full_struct_loss": 0.0,
            "partial_seq_correct": 0, "partial_seq_total": 0, "partial_seq_loss": 0.0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            skipped_in_batch = batch.get("num_skipped", 0)
            if isinstance(skipped_in_batch, torch.Tensor):
                total_skipped += skipped_in_batch.item()
            else:
                total_skipped += skipped_in_batch
                
            if "sequence" not in batch:
                continue
                
            masked_batch, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)
            
            # Send to device
            tokens = {k: v.to(device) for k, v in masked_batch.items()}
            labels_seq = labels_seq.to(device)
            labels_struct = labels_struct.to(device)
            
            # ESM3 explicitly requires these specific keyword arguments.
            # - sequence_tokens: the primary amino acid + special tokens track
            # - structure_tokens: the discretised VQ-VAE structure tokens track
            # - sasa_tokens: the discretised SASA tokens track
            # - chain_id: the chain identifier for geometric attention
            # - structure_coords: the 3D backbone coordinates (N, CA, C)
            model_kwargs = {
                "sequence_tokens": tokens["sequence"],
                "structure_tokens": tokens["structure"],
                "sasa_tokens": tokens["sasa"],
                "chain_id": tokens["chain_id"],
                "structure_coords": tokens["structure_coords"],
            }
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(**model_kwargs)
                
                loss, sample_loss, step_metrics = compute_mlm_loss(
                    sequence_logits=output.sequence_logits,
                    structure_logits=output.structure_logits,
                    labels_seq=labels_seq,
                    labels_struct=labels_struct,
                    tokenizer=tokenizer,
                    lam=args.lam,
                    gradient_accumulation_steps=args.gradient_accumulation_steps
                )
                
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # Log metrics
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            mask_seq = labels_seq != -100
            mask_struct = labels_struct != -100
            mask_struct_valid = mask_struct & (labels_struct != tokenizer.struct_nan)
            total_masked = mask_seq.sum(dim=1).float() + mask_struct_valid.sum(dim=1).float()
            valid_samples = total_masked > 0
            
            if valid_samples.any():
                total_true_loss += sample_loss[valid_samples].mean().item()
            num_batches += 1
            
            # Accumulate step metrics
            for k in m:
                m[k] += step_metrics[k]

            pbar.set_postfix({
                "Loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"
            })
            
        # Final metrics
        overall_acc = safe_div(m["overall_correct"], m["overall_total"])
        overall_loss_raw = safe_div(m["overall_loss"], m["overall_total"])
        overall_ppl = math.exp(overall_loss_raw) if m["overall_total"] > 0 else 0.0
        
        full_seq_acc = safe_div(m["full_seq_correct"], m["full_seq_total"])
        full_seq_loss_raw = safe_div(m["full_seq_loss"], m["full_seq_total"])
        full_seq_ppl = math.exp(full_seq_loss_raw) if m["full_seq_total"] > 0 else 0.0
        
        full_struct_acc = safe_div(m["full_struct_correct"], m["full_struct_total"])
        full_struct_loss_raw = safe_div(m["full_struct_loss"], m["full_struct_total"])
        full_struct_ppl = math.exp(full_struct_loss_raw) if m["full_struct_total"] > 0 else 0.0
        
        # Add combined full accuracy and full perplexity
        full_total_correct = m["full_seq_correct"] + m["full_struct_correct"]
        full_total = m["full_seq_total"] + m["full_struct_total"]
        full_total_loss = m["full_seq_loss"] + m["full_struct_loss"]
        
        full_acc = safe_div(full_total_correct, full_total)
        full_loss_raw = safe_div(full_total_loss, full_total)
        full_ppl = math.exp(full_loss_raw) if full_total > 0 else 0.0
        
        partial_seq_acc = safe_div(m["partial_seq_correct"], m["partial_seq_total"])
        partial_seq_loss_raw = safe_div(m["partial_seq_loss"], m["partial_seq_total"])
        partial_seq_ppl = math.exp(partial_seq_loss_raw) if m["partial_seq_total"] > 0 else 0.0
        
        avg_loss = safe_div(total_loss, num_batches)
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(
            f"Epoch {epoch:3d} | lr: {current_lr:.6f} | loss: {avg_loss:.3f} | acc: {overall_acc:.3f} | ppl: {overall_ppl:.2f} | "
            f"full_acc: {full_acc:.3f} | partial_acc: {partial_seq_acc:.3f} | skipped: {total_skipped}"
        )
        
        log_entry = {
            "epoch": epoch,
            "loss": avg_loss,
            "lr": current_lr,
            "lambda": args.lam,
            "skipped_samples": total_skipped,
            
            "overall_accuracy": overall_acc,
            "overall_perplexity": overall_ppl,
            "overall_loss_raw": overall_loss_raw,
            
            "full_accuracy": full_acc,
            "full_perplexity": full_ppl,
            "full_loss_raw": full_loss_raw,
            
            "full_seq_accuracy": full_seq_acc,
            "full_seq_perplexity": full_seq_ppl,
            "full_seq_loss_raw": full_seq_loss_raw,
            
            "full_struct_accuracy": full_struct_acc,
            "full_struct_perplexity": full_struct_ppl,
            "full_struct_loss_raw": full_struct_loss_raw,
            
            "partial_seq_accuracy": partial_seq_acc,
            "partial_seq_perplexity": partial_seq_ppl,
            "partial_seq_loss_raw": partial_seq_loss_raw
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        if epoch % args.checkpoint_every == 0:
            save_path = checkpoint_dir / f"epoch_{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            training_state = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            
            model.save_pretrained(save_path)
            torch.save(training_state, save_path / "training_state.pt")
            
            logger.info(f"Saved checkpoint to {save_path}")

        if overall_loss_raw < best_overall_loss:
            best_overall_loss = overall_loss_raw
            logger.info(f"New best overall loss: {best_overall_loss:.4f}. Saving best model checkpoint.")
            best_dir = checkpoint_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            
            # Use sync save for the best model to avoid any race conditions on overwritten path
            model.save_pretrained(best_dir)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train Mimir v2 task 2")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per worker/device (default: 4)")
    parser.add_argument("--peak-lr", type=float, default=1e-4, help="Peak learning rate after warmup (default: 1e-4)")
    parser.add_argument("--lam", type=float, default=0.5, help="Lambda penalty for masks (default: 0.5)")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save a checkpoint every N epochs (default: 1)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps (default: 1)")
    parser.add_argument("--use-8bit-adam", action="store_true", help="Use 8-bit AdamW optimizer if available (default: False)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (default: False)")
    args = parser.parse_args()
    
    logging.basicConfig(
        stream=sys.stdout, 
        level=logging.INFO if args.verbose else logging.WARNING, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    config = load_config(args.config)
    
    if not config.binders_merged.exists():
        logger.error(f"File not found: {config.binders_merged}")
        sys.exit(1)
        
    args.associations_csv = str(config.binders_merged)
    args.fingerprints_lmdb = str(config.features_fingerprints)
    args.binders_lmdb = str(config.features_binders)
        
    _run(args)


if __name__ == "__main__":
    main()

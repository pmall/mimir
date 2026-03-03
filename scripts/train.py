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

from mimir.config import load_config
from mimir.model import load_model
from mimir.dataset import MimirDataset, mimir_collate_fn, BucketBatchSampler
from mimir.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmdb").setLevel(logging.WARNING)

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
    
    cut_token_id = tokenizer.cut_seq
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
        "position_ids": batch["position_ids"],
        "attention_mask": batch["attention_mask"]
    }
    return masked_batch, labels_seq, labels_struct

# --- Main Training Logic ---

def safe_div(a: float, b: float) -> float: 
    return a / b if b > 0 else 0.0

def _run(args: argparse.Namespace) -> None:
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
    
    # Calculate steps first
    # Estimate total batches using math.ceil(len / batch_size) because bucket length is variable
    sampler = BucketBatchSampler(dataset=dataset, batch_size=args.batch_size, epoch=1)
    # len(sampler) returns estimated batches
    steps_per_epoch = len(sampler) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    logger.info(f"Estimated steps per epoch: {steps_per_epoch}, total steps: {total_steps}")
    
    # Checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 0
    latest_ckpt_path = None
    
    log_file = checkpoint_dir / "training_log.jsonl"
    
    if log_file.exists():
        with open(log_file, "r") as f:
            lines = f.readlines()
        if lines:
            last_log = json.loads(lines[-1])
            last_epoch = last_log.get("epoch", 0)
            if last_epoch > 0:
                ckpt_path = checkpoint_dir / f"epoch_{last_epoch}"
                if not ckpt_path.exists():
                    logger.error(f"Log indicates epoch {last_epoch} but checkpoint not found: {ckpt_path}")
                    sys.exit(1)
                latest_ckpt_path = str(ckpt_path)
                start_epoch = last_epoch
                logger.info(f"Resuming from epoch {start_epoch}")
    
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
        try:
            import bitsandbytes as bnb
            logger.info("Using 8-bit AdamW optimizer.")
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.peak_lr)
        except ImportError:
            logger.warning("bitsandbytes not found. Falling back to standard AdamW.")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.peak_lr)
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
    if latest_ckpt_path is not None:
        optimizer_path = os.path.join(latest_ckpt_path, "optimizer.pt")
        scheduler_path = os.path.join(latest_ckpt_path, "scheduler.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        logger.info(f"Resumed optimizer and scheduler from epoch {start_epoch}")
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
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
            "seq_correct": 0, "seq_total": 0, "seq_loss": 0.0,
            "struct_correct": 0, "struct_total": 0, "struct_loss": 0.0,
            "seq_only_correct": 0, "seq_only_total": 0, "seq_only_loss": 0.0,
            "has_struct_seq_correct": 0, "has_struct_seq_total": 0, "has_struct_seq_loss": 0.0
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
            
            model_kwargs = {
                "sequence_tokens": tokens["sequence"],
                "structure_tokens": tokens["structure"],
                "sasa_tokens": tokens["sasa"],
                "position_ids": tokens["position_ids"]
            }
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    output = model(**model_kwargs)
                
            loss_seq_per_token = criterion(output.sequence_logits.float().view(-1, output.sequence_logits.size(-1)), labels_seq.view(-1))
            loss_seq_per_token = loss_seq_per_token.view(labels_seq.size())
            
            loss_struct_per_token = criterion(output.structure_logits.float().view(-1, output.structure_logits.size(-1)), labels_struct.view(-1))
            loss_struct_per_token = loss_struct_per_token.view(labels_struct.size())
            
            mask_seq = labels_seq != -100
            mask_struct = labels_struct != -100
            
            num_masked_seq = mask_seq.sum(dim=1).float()
            num_masked_struct = mask_struct.sum(dim=1).float()
            total_masked = num_masked_seq + num_masked_struct
            
            sample_loss_seq = (loss_seq_per_token * mask_seq.float()).sum(dim=1)
            sample_loss_struct = (loss_struct_per_token * mask_struct.float()).sum(dim=1)
            
            # Unweighted loss per sample for tracking perplexity
            sample_loss = (sample_loss_seq + sample_loss_struct) / total_masked.clamp(min=1)
            
            # Apply log boost
            weight = args.lam * torch.log(1 + total_masked)
            boosted_loss = (sample_loss * weight)
            
            valid_samples = total_masked > 0
            if valid_samples.any():
                loss = boosted_loss[valid_samples].mean()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # Log metrics
            total_loss += loss.item() * args.gradient_accumulation_steps
            if valid_samples.any():
                total_true_loss += sample_loss[valid_samples].mean().item()
            num_batches += 1
            
            # Computations for detailed metrics
            with torch.no_grad():
                pred_seq = output.sequence_logits.argmax(dim=-1)
                pred_struct = output.structure_logits.argmax(dim=-1)
                
                correct_seq = (pred_seq == labels_seq) & mask_seq
                correct_struct = (pred_struct == labels_struct) & mask_struct
                
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
                        m["has_struct_seq_correct"] += seq_corr
                        m["has_struct_seq_total"] += nm_seq
                        m["has_struct_seq_loss"] += sample_loss_seq[i].item()
                        
                        m["struct_correct"] += struct_corr
                        m["struct_total"] += nm_struct
                        m["struct_loss"] += sample_loss_struct[i].item()
                    else:
                        m["seq_only_correct"] += seq_corr
                        m["seq_only_total"] += nm_seq
                        m["seq_only_loss"] += sample_loss_seq[i].item()

            pbar.set_postfix({
                "Loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"
            })
            
        # Final metrics
        overall_acc = safe_div(m["overall_correct"], m["overall_total"])
        overall_ppl = math.exp(safe_div(m["overall_loss"], m["overall_total"])) if m["overall_total"] > 0 else 0.0
        
        seq_acc = safe_div(m["has_struct_seq_correct"], m["has_struct_seq_total"])
        seq_ppl = math.exp(safe_div(m["has_struct_seq_loss"], m["has_struct_seq_total"])) if m["has_struct_seq_total"] > 0 else 0.0
        
        struct_acc = safe_div(m["struct_correct"], m["struct_total"])
        struct_ppl = math.exp(safe_div(m["struct_loss"], m["struct_total"])) if m["struct_total"] > 0 else 0.0
        
        seq_only_acc = safe_div(m["seq_only_correct"], m["seq_only_total"])
        seq_only_ppl = math.exp(safe_div(m["seq_only_loss"], m["seq_only_total"])) if m["seq_only_total"] > 0 else 0.0
        
        avg_loss = safe_div(total_loss, num_batches)
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(
            f"Epoch {epoch:3d} | lr: {current_lr:.6f} | loss: {avg_loss:.3f} | acc: {overall_acc:.3f} | ppl: {overall_ppl:.2f} | "
            f"seq_acc: {seq_acc:.3f} | struct_acc: {struct_acc:.3f} | seq_only_acc: {seq_only_acc:.3f} | skipped: {total_skipped}"
        )
        
        log_entry = {
            "epoch": epoch,
            "loss": avg_loss,
            "lr": current_lr,
            "lambda": args.lam,
            "skipped_samples": total_skipped,
            "overall_accuracy": overall_acc,
            "overall_perplexity": overall_ppl,
            "seq_accuracy": seq_acc,
            "seq_perplexity": seq_ppl,
            "struct_accuracy": struct_acc,
            "struct_perplexity": struct_ppl,
            "seq_only_accuracy": seq_only_acc,
            "seq_only_perplexity": seq_only_ppl
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        if epoch % args.checkpoint_every == 0:
            save_path = checkpoint_dir / f"epoch_{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            cut_embedding_keys = [
                "base_model.model.encoder.sequence_embed.cut_embedding.weight",
                "base_model.model.encoder.structure_tokens_embed.cut_embedding.weight",
                "base_model.model.encoder.sasa_embed.cut_embedding.weight",
            ]
            cut_embeddings = {k: model.state_dict()[k] for k in cut_embedding_keys}
            torch.save(cut_embeddings, save_path / "cut_embeddings.pt")
            
            threading.Thread(target=model.save_pretrained, args=(save_path,)).start()
            
            threading.Thread(target=torch.save, args=(optimizer.state_dict(), save_path / "optimizer.pt")).start()
            threading.Thread(target=torch.save, args=(scheduler.state_dict(), save_path / "scheduler.pt")).start()
            
            logger.info(f"Started async save for checkpoint to {save_path}")


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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (default: False)")
    args = parser.parse_args()
    
    logging.basicConfig(
        stream=sys.stdout, 
        level=logging.INFO if args.verbose else logging.WARNING, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    config = load_config(args.config)
    
    # Export environment variables for optimization right at runtime start
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if not config.binders_merged.exists():
        logger.error(f"File not found: {config.binders_merged}")
        sys.exit(1)
        
    args.associations_csv = str(config.binders_merged)
    args.fingerprints_lmdb = str(config.features_fingerprints)
    args.binders_lmdb = str(config.features_binders)
        
    _run(args)


if __name__ == "__main__":
    main()

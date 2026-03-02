import argparse
import logging
import os
import sys
import json
import random
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from mimir.model import load_model
    from mimir.dataset import MimirDataset, mimir_collate_fn, BucketBatchSampler
    from mimir.tokenizer import load_tokenizer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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

def r_dir(checkpoint_dir: str) -> Tuple[str | None, int]:
    if not os.path.exists(checkpoint_dir):
        return None, 0
    dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
    if not dirs:
        return None, 0
    epochs = [int(d.split("_")[1]) for d in dirs]
    latest_epoch = max(epochs)
    return os.path.join(checkpoint_dir, f"epoch_{latest_epoch}"), latest_epoch

def _run(args: argparse.Namespace) -> None:
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
    latest_ckpt_path, start_epoch = r_dir(checkpoint_dir)
    
    logger.info("Loading model...")
    model = load_model(latest_ckpt_path)
    model.to(device)
    model.train()
    
    # Optimizer & Scheduler
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            logger.info("Using 8-bit AdamW optimizer.")
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.peak_lr)
        except ImportError:
            logger.warning("bitsandbytes not found. Falling back to standard AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr)
    
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Resume opt & scheduler states
    if latest_ckpt_path is not None:
        state_path = os.path.join(latest_ckpt_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            logger.info(f"Resumed from epoch {start_epoch}")
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    log_file = Path("mimir_v2_training_log.jsonl")
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        sampler.set_epoch(epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: mimir_collate_fn(b, tokenizer),
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Epoch {epoch}/{args.epochs}")
        total_loss, total_true_loss = 0.0, 0.0
        num_batches = 0
        
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
            if not batch: # skipped
                continue
                
            masked_batch, labels_seq, labels_struct = apply_mlm_masking(batch, tokenizer)
            
            # Send to device
            tokens = {k: v.to(device) for k, v in masked_batch.items()}
            labels_seq = labels_seq.to(device)
            labels_struct = labels_struct.to(device)
            
            model_kwargs = {
                "sequence_tokens": tokens["sequence"],
                "structure_tokens": tokens["structure"],
                "sasa_tokens": tokens["sasa"]
            }
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            
        def safe_div(a: float, b: float) -> float: 
            return a / b if b > 0 else 0.0
        
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
            f"seq_acc: {seq_acc:.3f} | struct_acc: {struct_acc:.3f} | seq_only_acc: {seq_only_acc:.3f}"
        )
        
        log_entry = {
            "epoch": epoch,
            "loss": avg_loss,
            "lr": current_lr,
            "lambda": args.lam,
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
            
            # Save PEFT and embeddings via standard save? 
            # Wait! PEFT adapter can be saved, but we need the custom <cut> token embeddings.
            # Best is to just save the trainable parameters using state_dict
            trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
            torch.save({
                "model": trainable_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, save_path / "mimir_checkpoint.pt")
            logger.info(f"Saved checkpoint to {save_path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train Mimir v2 task 2")
    parser.add_argument("--associations-csv", type=str, required=True)
    parser.add_argument("--fingerprints-lmdb", type=str, required=True)
    parser.add_argument("--binders-lmdb", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per worker/device (default: 4)")
    parser.add_argument("--peak-lr", type=float, default=1e-4, help="Peak learning rate after warmup (default: 1e-4)")
    parser.add_argument("--lam", type=float, default=0.5, help="Lambda penalty for masks (default: 0.5)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save a checkpoint every N epochs (default: 5)")
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
    
    if not os.path.exists(args.associations_csv):
        logger.error(f"File not found: {args.associations_csv}")
        sys.exit(1)
        
    _run(args)


if __name__ == "__main__":
    main()

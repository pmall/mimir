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
from pathlib import Path
from typing import Any

# Fix segment fragmentation to avoid OOMs on long runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Ensure stdout is not buffered so we easily see logs in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- Masking Strategy ---


def apply_mlm_masking(batch: dict, tokenizer: Any) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Applies MLM masking to the binder region (after chainbreak).

    Sequence and structure tracks are masked independently with separate
    uniform draws in [0.25, 0.75]. Structure is only masked when the binder
    has real structure tokens (not all MASK), i.e. Case A.

    Returns:
        (masked_batch, labels_seq, labels_struct)
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

        # Sequence: always masked with independent rate
        mask_rate_seq = random.uniform(0.25, 0.75)
        num_mask_seq = max(1, int(round(binder_len * mask_rate_seq)))
        mask_indices_seq = random.sample(range(binder_start, binder_end), num_mask_seq)

        for idx in mask_indices_seq:
            labels_seq[i, idx] = seq[i, idx].item()
            seq[i, idx] = tokenizer.seq_mask

        # Structure: masked only if binder has real structure (Case A)
        struct_binder = struct[i, binder_start:binder_end]

        if not torch.all(struct_binder == tokenizer.struct_mask):
            mask_rate_struct = random.uniform(0.25, 0.75)
            num_mask_struct = max(1, int(round(binder_len * mask_rate_struct)))
            mask_indices_struct = random.sample(range(binder_start, binder_end), num_mask_struct)

            for idx in mask_indices_struct:
                labels_struct[i, idx] = struct[i, idx].item()
                struct[i, idx] = tokenizer.struct_mask

    masked_batch = {
        "sequence": seq,
        "structure": struct,
        "sasa": batch["sasa"],
        "chain_id": batch["chain_id"],
        "structure_coords": batch["structure_coords"],
        "sequence_id": batch["sequence_id"],
    }

    return masked_batch, labels_seq, labels_struct


def compute_mlm_loss(
    sequence_logits: torch.Tensor,
    structure_logits: torch.Tensor,
    labels_seq: torch.Tensor,
    labels_struct: torch.Tensor,
    tokenizer: Any,
    lam: float,
    gradient_accumulation_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Computes the weighted Masked Language Modeling loss and granular metrics.

    Args:
        sequence_logits: (B, L, V_seq)
        structure_logits: (B, L, V_struct)
        labels_seq: (B, L) with -100 for ignored tokens
        labels_struct: (B, L) with -100 for ignored tokens and 2246 for NaN tokens
        lam: lambda penalty weight for number of masked tokens
        gradient_accumulation_steps: divide final loss by this factor

    Returns:
        loss: scalar loss for backpropagation (divided by accumulation steps)
        sample_loss: unweighted per-sample loss tensor (B,) for perplexity tracking
        metrics: dictionary of raw correct/total/loss sums for metric tracking
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    device = sequence_logits.device

    loss_seq_per_token = criterion(
        sequence_logits.float().view(-1, sequence_logits.size(-1)),
        labels_seq.view(-1),
    ).view(labels_seq.size())

    loss_struct_per_token = criterion(
        structure_logits.float().view(-1, structure_logits.size(-1)),
        labels_struct.view(-1),
    ).view(labels_struct.size())

    mask_seq = labels_seq != -100
    mask_struct = labels_struct != -100

    # Exclude positions where the ground-truth structure token is 2246 (nan coords).
    # These positions were masked in the input but are unanswerable — the model
    # should not be penalised for them.
    mask_struct_valid = mask_struct & (labels_struct != tokenizer.struct_nan)

    num_masked_seq = mask_seq.sum(dim=1).float()
    num_masked_struct = mask_struct_valid.sum(dim=1).float()
    total_masked = num_masked_seq + num_masked_struct

    sample_loss_seq = (loss_seq_per_token * mask_seq.float()).sum(dim=1)
    sample_loss_struct = (loss_struct_per_token * mask_struct_valid.float()).sum(dim=1)

    # Average per-token loss per sample (unweighted, for perplexity tracking)
    sample_loss = (sample_loss_seq + sample_loss_struct) / total_masked.clamp(min=1)

    # Log boost: samples with more masked tokens get higher loss weight.
    # This pushes the model to learn generation from scratch (fully masked).
    # torch.log is the natural logarithm (ln), intentional per spec.
    weight = 1.0 + lam * torch.log(1 + total_masked)
    boosted_loss = sample_loss * weight

    valid_samples = total_masked > 0

    if valid_samples.any():
        loss = boosted_loss[valid_samples].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    loss = loss / gradient_accumulation_steps

    # Detailed metrics for epoch-level reporting
    m = _compute_detailed_metrics(
        labels_seq, labels_struct,
        sequence_logits, structure_logits,
        mask_seq, mask_struct_valid,
        num_masked_seq, num_masked_struct,
        sample_loss_seq, sample_loss_struct,
    )

    return loss, sample_loss, m


def _compute_detailed_metrics(
    labels_seq: torch.Tensor,
    labels_struct: torch.Tensor,
    sequence_logits: torch.Tensor,
    structure_logits: torch.Tensor,
    mask_seq: torch.Tensor,
    mask_struct_valid: torch.Tensor,
    num_masked_seq: torch.Tensor,
    num_masked_struct: torch.Tensor,
    sample_loss_seq: torch.Tensor,
    sample_loss_struct: torch.Tensor,
) -> dict[str, float]:
    """Accumulates per-sample accuracy / loss counts into metric buckets.

    Splits samples into "full" (has structure supervision) and "partial"
    (sequence-only supervision) for separate tracking.
    """
    m: dict[str, float] = {
        "overall_correct": 0, "overall_total": 0, "overall_loss": 0.0,
        "full_seq_correct": 0, "full_seq_total": 0, "full_seq_loss": 0.0,
        "full_struct_correct": 0, "full_struct_total": 0, "full_struct_loss": 0.0,
        "partial_seq_correct": 0, "partial_seq_total": 0, "partial_seq_loss": 0.0,
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

    return m


# --- Resume ---


def _resolve_resume_state(
    checkpoint_dir: Path,
) -> tuple[int, str | None, dict | None, float]:
    """Scans the training log to find the last checkpoint and best loss.

    Returns:
        (start_epoch, latest_ckpt_path, training_state, best_overall_loss)
    """
    start_epoch = 0
    latest_ckpt_path = None
    training_state = None
    best_overall_loss = float("inf")

    log_file = checkpoint_dir / "training_log.jsonl"

    if not log_file.exists():
        return start_epoch, latest_ckpt_path, training_state, best_overall_loss

    with open(log_file, "r") as f:
        lines = f.readlines()

    if not lines:
        return start_epoch, latest_ckpt_path, training_state, best_overall_loss

    # Scan all log lines for the best loss seen in any previous run
    for line in lines:
        log = json.loads(line)
        if "overall_loss_raw" in log and log["overall_loss_raw"] < best_overall_loss:
            best_overall_loss = log["overall_loss_raw"]

    # Resume from the last logged epoch (which always has a matching checkpoint,
    # because the log line is written inside the checkpoint-save block)
    last_log = json.loads(lines[-1])
    last_epoch = last_log.get("epoch", 0)

    if last_epoch <= 0:
        return start_epoch, latest_ckpt_path, training_state, best_overall_loss

    ckpt_path = checkpoint_dir / f"epoch_{last_epoch}"

    if not ckpt_path.exists():
        logger.error(f"Log indicates epoch {last_epoch} but checkpoint not found: {ckpt_path}")
        sys.exit(1)

    training_state_path = ckpt_path / "training_state.pt"

    if not training_state_path.exists():
        logger.error(
            f"Log indicates epoch {last_epoch} but no training_state.pt "
            f"found in {ckpt_path}. Cannot guarantee safe resume."
        )
        sys.exit(1)

    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)

    # Cross-check: the epoch in training_state.pt must match the log
    saved_epoch = training_state.get("epoch")
    if saved_epoch != last_epoch:
        logger.error(
            f"Integrity Error: training_log says epoch {last_epoch}, "
            f"but {training_state_path} says epoch {saved_epoch}."
        )
        sys.exit(1)

    start_epoch = saved_epoch
    latest_ckpt_path = str(ckpt_path)
    logger.info(f"Resuming safely from epoch {start_epoch}")

    return start_epoch, latest_ckpt_path, training_state, best_overall_loss


# --- Optimizer & Scheduler ---


def _build_optimizer(
    model: torch.nn.Module,
    peak_lr: float,
    use_8bit_adam: bool,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    """Creates the optimizer and returns (optimizer, trainable_params)."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if use_8bit_adam:
        if not HAS_BNB:
            logger.error("use_8bit_adam requested but bitsandbytes is not installed.")
            sys.exit(1)
        logger.info("Using 8-bit AdamW optimizer.")
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=peak_lr)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=peak_lr)

    return optimizer, trainable_params


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    peak_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup (5% of total steps) then cosine decay to 1e-5."""
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = 1e-5 / peak_lr
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- Epoch Metrics ---


def safe_div(a: float, b: float) -> float:
    """Safe division returning 0.0 when denominator is zero."""
    return a / b if b > 0 else 0.0


def _compute_epoch_metrics(m: dict[str, float]) -> dict[str, float]:
    """Derives per-token loss, accuracy, and perplexity from raw counters."""
    overall_acc = safe_div(m["overall_correct"], m["overall_total"])
    overall_loss_raw = safe_div(m["overall_loss"], m["overall_total"])
    overall_ppl = math.exp(overall_loss_raw) if m["overall_total"] > 0 else 0.0

    full_seq_acc = safe_div(m["full_seq_correct"], m["full_seq_total"])
    full_seq_loss_raw = safe_div(m["full_seq_loss"], m["full_seq_total"])
    full_seq_ppl = math.exp(full_seq_loss_raw) if m["full_seq_total"] > 0 else 0.0

    full_struct_acc = safe_div(m["full_struct_correct"], m["full_struct_total"])
    full_struct_loss_raw = safe_div(m["full_struct_loss"], m["full_struct_total"])
    full_struct_ppl = math.exp(full_struct_loss_raw) if m["full_struct_total"] > 0 else 0.0

    full_total_correct = m["full_seq_correct"] + m["full_struct_correct"]
    full_total = m["full_seq_total"] + m["full_struct_total"]
    full_total_loss = m["full_seq_loss"] + m["full_struct_loss"]

    full_acc = safe_div(full_total_correct, full_total)
    full_loss_raw = safe_div(full_total_loss, full_total)
    full_ppl = math.exp(full_loss_raw) if full_total > 0 else 0.0

    partial_seq_acc = safe_div(m["partial_seq_correct"], m["partial_seq_total"])
    partial_seq_loss_raw = safe_div(m["partial_seq_loss"], m["partial_seq_total"])
    partial_seq_ppl = math.exp(partial_seq_loss_raw) if m["partial_seq_total"] > 0 else 0.0

    return {
        "overall_acc": overall_acc,
        "overall_loss_raw": overall_loss_raw,
        "overall_ppl": overall_ppl,

        "full_acc": full_acc,
        "full_loss_raw": full_loss_raw,
        "full_ppl": full_ppl,

        "full_seq_acc": full_seq_acc,
        "full_seq_loss_raw": full_seq_loss_raw,
        "full_seq_ppl": full_seq_ppl,

        "full_struct_acc": full_struct_acc,
        "full_struct_loss_raw": full_struct_loss_raw,
        "full_struct_ppl": full_struct_ppl,

        "partial_seq_acc": partial_seq_acc,
        "partial_seq_loss_raw": partial_seq_loss_raw,
        "partial_seq_ppl": partial_seq_ppl,
    }


def _log_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_loss: float,
    current_lr: float,
    total_skipped: int,
    em: dict[str, float],
) -> None:
    """Prints a readable epoch summary to stdout."""
    logger.info(
        f"\nEpoch {epoch}/{total_epochs}  —  LR: {current_lr:.6f}  Boosted Loss: {avg_loss:.4f}  Skipped: {total_skipped}\n"
        f"  Overall          Loss: {em['overall_loss_raw']:.4f}  PPL: {em['overall_ppl']:8.2f}  Acc: {em['overall_acc']:.3f}\n"
        f"  Full (seq+struct) Loss: {em['full_loss_raw']:.4f}  PPL: {em['full_ppl']:8.2f}  Acc: {em['full_acc']:.3f}\n"
        f"    Seq             Loss: {em['full_seq_loss_raw']:.4f}  PPL: {em['full_seq_ppl']:8.2f}  Acc: {em['full_seq_acc']:.3f}\n"
        f"    Struct          Loss: {em['full_struct_loss_raw']:.4f}  PPL: {em['full_struct_ppl']:8.2f}  Acc: {em['full_struct_acc']:.3f}\n"
        f"  Partial (seq)     Loss: {em['partial_seq_loss_raw']:.4f}  PPL: {em['partial_seq_ppl']:8.2f}  Acc: {em['partial_seq_acc']:.3f}"
    )


def _build_log_entry(
    epoch: int,
    avg_loss: float,
    current_lr: float,
    lam: float,
    total_skipped: int,
    em: dict[str, float],
) -> dict[str, Any]:
    """Constructs the JSONL log entry for one epoch."""
    return {
        "epoch": epoch,
        "loss": avg_loss,
        "lr": current_lr,
        "lambda": lam,
        "skipped_samples": total_skipped,

        "overall_accuracy": em["overall_acc"],
        "overall_perplexity": em["overall_ppl"],
        "overall_loss_raw": em["overall_loss_raw"],

        "full_accuracy": em["full_acc"],
        "full_perplexity": em["full_ppl"],
        "full_loss_raw": em["full_loss_raw"],

        "full_seq_accuracy": em["full_seq_acc"],
        "full_seq_perplexity": em["full_seq_ppl"],
        "full_seq_loss_raw": em["full_seq_loss_raw"],

        "full_struct_accuracy": em["full_struct_acc"],
        "full_struct_perplexity": em["full_struct_ppl"],
        "full_struct_loss_raw": em["full_struct_loss_raw"],

        "partial_seq_accuracy": em["partial_seq_acc"],
        "partial_seq_perplexity": em["partial_seq_ppl"],
        "partial_seq_loss_raw": em["partial_seq_loss_raw"],
    }


# --- Checkpointing ---


def _save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    save_path: Path,
) -> None:
    """Saves adapter weights and training state to the given directory.

    Always produces the same files:
        - adapter_config.json + adapter_model.safetensors (from PEFT)
        - training_state.pt (epoch, optimizer, scheduler)

    Used by both epoch checkpoints and best-model saves so the
    checkpoint format is guaranteed identical.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    torch.save(
        {"epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        save_path / "training_state.pt",
    )

    logger.info(f"Saved checkpoint to {save_path}")


def _save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    checkpoint_dir: Path,
    log_entry: dict[str, Any],
) -> None:
    """Saves an epoch checkpoint then appends the JSONL log line.

    The log is written AFTER the checkpoint files to guarantee crash safety:
    if we crash mid-save, the log won't reference a half-written checkpoint.
    """
    _save_model_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir / f"epoch_{epoch}")

    log_file = checkpoint_dir / "training_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# --- Main Training Logic ---


def _run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Flash Attention 2 / SDPA
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("Flash Attention 2 / SDPA prioritized.")

    # Data
    logger.info("Loading tokenizer and configuring dataloader...")
    tokenizer = load_tokenizer()

    dataset = MimirDataset(
        associations_csv=args.associations_csv,
        fingerprints_lmdb=args.fingerprints_lmdb,
        binders_lmdb=args.binders_lmdb,
        tokenizer=tokenizer,
    )
    logger.info(f"Total samples: {len(dataset)}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Sampler & step counts
    sampler = BucketBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        cache_path=checkpoint_dir,
        epoch=1,
    )

    steps_per_epoch = len(sampler) // args.gradient_accumulation_steps
    logger.info(f"Estimated steps per epoch: {steps_per_epoch}")

    # Resume
    start_epoch, latest_ckpt_path, training_state, best_overall_loss = (
        _resolve_resume_state(checkpoint_dir)
    )

    # Remaining steps for the scheduler — accounts for epochs already completed
    remaining_epochs = args.epochs - start_epoch
    total_steps = remaining_epochs * steps_per_epoch
    logger.info(f"Remaining epochs: {remaining_epochs}, total scheduler steps: {total_steps}")

    # Model
    logger.info("Loading model...")
    model = load_model(latest_ckpt_path)
    model.to(device)
    model.train()

    logger.info("Compiling model (this may take a moment)...")
    model = torch.compile(model, mode="reduce-overhead")

    # Optimizer & scheduler
    optimizer, trainable_params = _build_optimizer(model, args.peak_lr, args.use_8bit_adam)
    scheduler = _build_scheduler(optimizer, total_steps, args.peak_lr)

    if training_state is not None:
        if "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        if "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])
        logger.info(f"Resumed optimizer and scheduler from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        sampler.set_epoch(epoch)

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: mimir_collate_fn(b, tokenizer),
            num_workers=args.num_workers,
            persistent_workers=False,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        logger.info(f"Epoch {epoch}/{args.epochs}")

        total_loss = 0.0
        num_batches = 0
        total_skipped = 0

        m = {
            "overall_correct": 0, "overall_total": 0, "overall_loss": 0.0,
            "full_seq_correct": 0, "full_seq_total": 0, "full_seq_loss": 0.0,
            "full_struct_correct": 0, "full_struct_total": 0, "full_struct_loss": 0.0,
            "partial_seq_correct": 0, "partial_seq_total": 0, "partial_seq_loss": 0.0,
        }

        # Zero gradients at epoch start to prevent leaking leftover gradients
        # from the previous epoch when total batches is not divisible by
        # gradient_accumulation_steps.
        optimizer.zero_grad()

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

            tokens = {k: v.to(device) for k, v in masked_batch.items()}
            labels_seq = labels_seq.to(device)
            labels_struct = labels_struct.to(device)

            model_kwargs = {
                "sequence_tokens": tokens["sequence"],
                "structure_tokens": tokens["structure"],
                "sasa_tokens": tokens["sasa"],
                "chain_id": tokens["chain_id"],
                "structure_coords": tokens["structure_coords"],
                "sequence_id": tokens["sequence_id"],
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
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )

            loss.backward()

            # Increment valid batches processed. Using num_batches instead of
            # `step` ensures we don't inappropriately count skipped/empty batches
            num_batches += 1

            if num_batches % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps

            for k in m:
                m[k] += step_metrics[k]

            pbar.set_postfix({"Loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

        # Process any remaining accumulated gradients at the end of the epoch
        # to ensure no valid training samples are discarded if the total number
        # of batches isn't perfectly divisible by gradient_accumulation_steps.
        if num_batches > 0 and num_batches % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Epoch summary
        avg_loss = safe_div(total_loss, num_batches)
        current_lr = scheduler.get_last_lr()[0]
        em = _compute_epoch_metrics(m)

        _log_epoch_summary(epoch, args.epochs, avg_loss, current_lr, total_skipped, em)

        log_entry = _build_log_entry(epoch, avg_loss, current_lr, args.lam, total_skipped, em)

        # Epoch checkpoint: the JSONL log line is written inside
        # _save_epoch_checkpoint, only when a checkpoint is actually saved.
        # The resume logic reads the last log line and loads the matching
        # epoch_N/ directory — writing a log without a checkpoint would
        # break resume.
        if epoch % args.checkpoint_every == 0:
            _save_epoch_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, log_entry)

        # Best model: same checkpoint contents as a regular epoch save
        # (via _save_model_checkpoint) plus a best_model.json for reference.
        if em["overall_loss_raw"] < best_overall_loss:
            best_overall_loss = em["overall_loss_raw"]
            logger.info(f"New best overall loss: {best_overall_loss:.4f}. Saving best model.")

            best_dir = checkpoint_dir / "best_model"
            _save_model_checkpoint(model, optimizer, scheduler, epoch, best_dir)

            with open(best_dir / "best_model.json", "w") as f:
                json.dump(log_entry, f, indent=2)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="Train Mimir v2 task 2")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per worker/device (default: 32)")
    parser.add_argument("--peak-lr", type=float, default=1e-4, help="Peak learning rate after warmup (default: 1e-4)")
    parser.add_argument("--lam", type=float, default=1.0, help="Lambda penalty for masks (default: 1.0)")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save a checkpoint every N epochs (default: 1)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps (default: 4)")
    parser.add_argument("--use-8bit-adam", action="store_true", help="Use 8-bit AdamW optimizer if available (default: False)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (default: False)")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)

    if not config.binders_merged.exists():
        logger.error(f"File not found: {config.binders_merged}")
        sys.exit(1)

    # Extract dataset paths from config onto args so _run is decoupled from
    # the config system. This follows the project pattern where main() bridges
    # config and the execution function via named args.
    args.associations_csv = str(config.binders_merged)
    args.fingerprints_lmdb = str(config.features_fingerprints)
    args.binders_lmdb = str(config.features_binders)

    _run(args)


if __name__ == "__main__":
    main()

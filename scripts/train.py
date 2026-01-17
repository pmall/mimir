
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model, LoraConfig

from mimir.dataset import PeptideDataset, create_dynamic_collate_fn
from mimir.tokenizer import AminoAcidTokenizer
from esm.models.esm3 import ESM3

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
    
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        return

    # Extract unique targets to build vocabulary
    import csv
    targets = set()
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.add(row["target"])
    print(f"Found {len(targets)} unique targets.")

    # Initialize Tokenizer with targets
    tokenizer = AminoAcidTokenizer(targets=list(targets))
    
    # Initialize Dataset
    dataset = PeptideDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Initialize Dynamic Collator with Masking
    # We pass both pad_idx and mask_idx to handle padding and masking logic
    collate_fn = create_dynamic_collate_fn(
        pad_idx=dataset.tokenizer.pad_idx,
        mask_idx=dataset.tokenizer.mask_token_id
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # 3. Model Initialization
    # -----------------------
    print("Loading ESM-3 model...")
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model.to(device)
    
    # CRITICAL: Resize Embeddings
    # ---------------------------
    # We must resizing the model's embedding tables to accommodate the new target tokens.
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    
    # Loss Function:
    # We use CrossEntropyLoss with reduction='mean'.
    # IMPORTANT: ignore_index=-100 ensures we only calculate loss on MASKED tokens.
    # The 'mean' reduction then automatically divides by the number of valid (masked) tokens,
    # satisfying the normalization requirement (Sum of Loss / N_masks).
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    # 6. Training Execution
    # ---------------------
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader):
            # Move inputs to device
            tokens = batch["tokens"].to(device)   # Input: Masked sequences
            labels = batch["labels"].to(device)   # Target: Original IDs at masked pos, -100 otherwise
            
            optimizer.zero_grad()
            
            try:
                # Forward Pass
                # We simply pass the token sequence. 
                # ESM-3 returns an output object containing 'sequence_logits'.
                output = model(sequence_tokens=tokens)
                
                # Compute Loss
                # Output shape: [Batch, Length, Vocab]
                logits = output.sequence_logits
                
                # Flatten logits and labels for CrossEntropy
                # Logits: [B*L, Vocab]
                # Labels: [B*L]
                loss = criterion(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in training step: {e}")
                # Optional: print full traceback for debugging
                import traceback
                traceback.print_exc()
                break
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically
        save_path = f"checkpoints/epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()
    
    train(args)

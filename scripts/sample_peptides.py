"""
Script to sample peptides from a fine-tuned ESM-3 model.
Uses MaskGit-style Parallel Iterative Decoding (Cosine Schedule).
"""

import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.tokenization import EsmSequenceTokenizer
from peft import PeftModel

# Import from mimir package
# Ensure mimir is in path if running from root
sys.path.append(os.getcwd())
from mimir.tokenizer import AminoAcidTokenizer
from mimir.model_utils import resize_esm3_tokens

def generate_peptide(model, tokenizer, target_id, length, temperature=1.0, top_n=0.25, device="cuda"):
    """
    Generate a peptide sequence using Top-N Adaptive Unmasking.
    
    Strategy:
    1. Logic operates entirely in Temperature-scaled probability space.
    2. At each step, unmask the 'Top N' most confident positions.
    3. N is calculated based on top_n parameter:
       - If float (0<n<1): ceil(remaining_masks * top_n)
       - If int (n>=1): n
    """
    mask_idx = tokenizer.mask_token_id
    
    # 1. Initialize Sequence: [TARGET, BOS, MASK * length, EOS]
    input_ids = [target_id, tokenizer.bos_idx] + [mask_idx] * length + [tokenizer.eos_idx]
    input_tensor = torch.tensor([input_ids], device=device) # [1, L+3]
    
    # Indices corresponding to the generated peptide (skipping Target & BOS)
    peptide_indices = list(range(2, length + 2))
    
    model.eval()
    
    masked_indices = set(peptide_indices)
    
    while masked_indices:
        # Determine step size (N) for this iteration
        if isinstance(top_n, float):
            # Proportional decreasing schedule
            # e.g. 0.25 * 20 -> 5. 0.25 * 5 -> 2.
            n_step = math.ceil(len(masked_indices) * top_n)
            n_step = max(1, n_step) # At least 1
        else:
            # Fixed step size
            n_step = top_n

        with torch.no_grad():
            output = model(sequence_tokens=input_tensor)
            logits = output.sequence_logits.float() # [1, SeqLen, Vocab]
            
            # Mask out targets and specials from being selected
            logits[..., tokenizer.base_vocab_size:] = float('-inf')
            for special in [tokenizer.pad_idx, tokenizer.bos_idx, tokenizer.eos_idx, tokenizer.mask_token_id]:
                logits[..., special] = float('-inf')
            
            # 1. Selection Phase (T=1.0): "Where are we most confident?"
            probs_raw = F.softmax(logits, dim=-1)

            candidates = []
            for pos in masked_indices:
                p_raw = probs_raw[0, pos]
                # Intrinsic Model Confidence (Max Probability)
                # We do NOT sample here. We just ask "How sure represent you?"
                max_prob = p_raw.max().item()
                candidates.append({'pos': pos, 'confidence': max_prob})
            
            # Rank by Intrinsic Confidence (Structure)
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Select Top N to unmask
            actual_n = min(n_step, len(masked_indices))
            to_unmask_candidates = candidates[:actual_n]
            
            # 2. Sampling Phase (T=User): "What are we dreaming?"
            # For the SELECTED positions, we sample using temperature.
            for item in to_unmask_candidates:
                pos = item['pos']
                
                # Apply Temperature to Logits
                logits_pos = logits[0, pos] / temperature
                p_temp = F.softmax(logits_pos, dim=-1)
                
                # Sample Token
                token = torch.multinomial(p_temp, 1).item()
                
                # Commit
                input_tensor[0, pos] = token
                masked_indices.remove(pos)
    
    # Decode
    generated_ids = input_tensor[0].tolist()
    seq_ids = generated_ids[2:-1] 
    sequence = tokenizer.decode(seq_ids)
    
    return sequence

def main():
    parser = argparse.ArgumentParser(description="Sample peptides from fine-tuned ESM-3 using MaskGit")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--targets", type=str, required=True, help="Comma-separated list of target names")
    parser.add_argument("--min_size", type=int, default=10, help="Minimum peptide length")
    parser.add_argument("--max_size", type=int, default=20, help="Maximum peptide length")
    parser.add_argument("--num_peptides", type=int, default=1, help="Number of peptides per target/length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_n", type=str, default="0.25", help="Batch size (int > 1 or float 0-1 for percentage). Default 0.25")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Parse top_n
    try:
        if '.' in args.top_n:
            top_n_val = float(args.top_n)
            if not (0 < top_n_val <= 1.0):
                 raise ValueError
        else:
            top_n_val = int(args.top_n)
            if top_n_val < 1:
                raise ValueError
    except ValueError:
        print(f"Error: Invalid value for --top_n: {args.top_n}. Must be float (0.0-1.0) or int (>=1).")
        sys.exit(1)
    
    # 1. Load Vocab
    vocab_path = os.path.join(args.checkpoint_path, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Error: vocab.json not found in {args.checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r') as f:
        targets_vocab = json.load(f)
        
    # 2. Initialize Tokenizer
    tokenizer = AminoAcidTokenizer(targets=targets_vocab)
    print(f"Tokenizer initialized. Vocab size: {tokenizer.vocab_size}")
    
    # 3. Load Model
    print("Loading ESM-3 base model...")
    base_model = ESM3.from_pretrained("esm3_sm_open_v1")
    
    print("Resizing embeddings...")
    resize_esm3_tokens(base_model, tokenizer.vocab_size)
    # Check if vocabulary size matches
    if base_model.encoder.sequence_embed.num_embeddings != tokenizer.vocab_size:
        print(f"Error: Model embedding size ({base_model.encoder.sequence_embed.num_embeddings}) does not match tokenizer size ({tokenizer.vocab_size}).")
        sys.exit(1)
    
    print(f"Loading LoRA adapter from {args.checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    model.to(args.device)
    model.eval()
    
    # 4. Generation Loop
    target_list = [t.strip() for t in args.targets.split(",")]
    
    print("-" * 60)
    print(f"Generating peptides for {len(target_list)} targets.")
    print(f"Length range: {args.min_size}-{args.max_size}")
    print(f"Temp: {args.temperature} | Top-N: {top_n_val}")
    print("-" * 60)
    
    for target_name in target_list:
        try:
            target_id = tokenizer.get_target_id(target_name)
        except ValueError:
            print(f"Warning: Target '{target_name}' not found. Skipping.")
            continue
            
        print(f"\nTarget: {target_name} (ID: {target_id})")
        
        for length in range(args.min_size, args.max_size + 1):
            print(f"  Length {length}:")
            for i in range(args.num_peptides):
                seq = generate_peptide(
                    model=model, 
                    tokenizer=tokenizer, 
                    target_id=target_id, 
                    length=length, 
                    temperature=args.temperature,
                    top_n=top_n_val,
                    device=args.device
                )
                print(f"    {i+1}: {seq}")

if __name__ == "__main__":
    main()

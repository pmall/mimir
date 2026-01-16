
from typing import List
from esm.tokenization import EsmSequenceTokenizer

class AminoAcidTokenizer:
    """
    Tokenizer for Amino Acid sequences using ESM-3's vocabulary.
    
    This wrapper ensures consistent encoding/decoding and handles
    sequence truncation/padding logic required for fixed-size or dynamic batching.
    
    Attributes:
        tokenizer (EsmSequenceTokenizer): The underlying ESM-3 tokenizer.
        pad_idx (int): Index for the padding token.
        bos_idx (int): Index for the Beginning Of Sequence token.
        eos_idx (int): Index for the End Of Sequence token.
    """

    def __init__(self, targets: List[str] = None):
        self.tokenizer = EsmSequenceTokenizer()
        self.pad_idx = self.tokenizer.pad_token_id
        self.bos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        
        # Standard vocab size
        self.base_vocab_size = self.tokenizer.vocab_size
        
        # Target tokens
        self.target_to_id = {}
        self.id_to_target = {}
        
        if targets:
            # Sort for deterministic ID assignment
            for i, target in enumerate(sorted(set(targets))):
                # Assign IDs starting after base vocab
                token_id = self.base_vocab_size + i
                self.target_to_id[target] = token_id
                self.id_to_target[token_id] = target
                
    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.target_to_id)

    def get_target_id(self, target: str) -> int:
        if target not in self.target_to_id:
            raise ValueError(f"Unknown target: {target}")
        return self.target_to_id[target]

    def encode(self, sequence: str, max_length: int = None) -> List[int]:
        """
        Encode a sequence into token IDs.
        Includes BOS and EOS tokens automatically by the underlying tokenizer.
        """
        # EsmSequenceTokenizer.encode adds BOS and EOS by default?
        # Based on explore output: [0, 20, ..., 2]. 0 is likely BOS, 2 is EOS.
        ids = self.tokenizer.encode(sequence)
        
        if max_length is None:
            return ids
            
        if len(ids) > max_length:
            # Simple truncation. 
            # Ideally we keep valid tokens.
            ids = ids[:max_length]
        
        if len(ids) < max_length:
            ids = ids + [self.pad_idx] * (max_length - len(ids))
            
        return ids

    def decode(self, tokens: List[int]) -> str:
        # Handle custom tokens if present in the list?
        # For now, filter them out before decoding with base tokenizer, or handle errors.
        base_tokens = [t for t in tokens if t < self.base_vocab_size]
        return self.tokenizer.decode(base_tokens)

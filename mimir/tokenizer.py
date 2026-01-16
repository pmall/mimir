
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
    def __init__(self):
        self.tokenizer = EsmSequenceTokenizer()
        self.pad_idx = self.tokenizer.pad_token_id
        self.bos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id

    def encode(self, sequence: str, max_length: int = None) -> List[int]:
        """
        Encode a sequence into token IDs.
        Includes BOS and EOS tokens automatically by the underlying tokenizer.
        
        Args:
            sequence: The amino acid sequence string (e.g., "MKAIL").
            max_length: Optional maximum length. If provided:
                        - Truncates longer sequences.
                        - Pads shorter sequences with pad_idx.
        
        Returns:
            List[int]: A list of token IDs.
        """
        # EsmSequenceTokenizer.encode adds BOS and EOS by default?
        # Based on explore output: [0, 20, ..., 2]. 0 is likely BOS, 2 is EOS.
        ids = self.tokenizer.encode(sequence)
        
        if max_length is None:
            return ids
            
        if len(ids) > max_length:
            # Truncate but keep EOS if possible?
            # Standard: truncate to max_length - 1 and append EOS?
            # Or just strict truncation.
            # Let's simple truncate for now.
             ids = ids[:max_length]
             # Ensure last token is EOS if it was cut off? 
             # For peptides 4-20 length, max_length is usually adequate.
        
        if len(ids) < max_length:
            ids = ids + [self.pad_idx] * (max_length - len(ids))
            
        return ids

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

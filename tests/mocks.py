"""
Mocks for Mimir v2 tests and sanity checks.
"""

import torch
import torch.nn as nn

from mimir.tokenizer import CUT_TOKEN_ID_SEQ, CUT_TOKEN_ID_STRUCT

SEQ_VOCAB_SIZE = CUT_TOKEN_ID_SEQ + 1      # 65
STRUCT_VOCAB_SIZE = CUT_TOKEN_ID_STRUCT + 1  # 4101
MOCK_HIDDEN = 16

class _MockOutput:
    def __init__(self, sequence_logits: torch.Tensor, structure_logits: torch.Tensor) -> None:
        self.sequence_logits = sequence_logits
        self.structure_logits = structure_logits

class MockEsm3(nn.Module):
    def __init__(self, vocab_seq: int = SEQ_VOCAB_SIZE, vocab_struct: int = STRUCT_VOCAB_SIZE) -> None:
        super().__init__()
        self.seq_embed = nn.Embedding(vocab_seq, MOCK_HIDDEN)
        self.struct_embed = nn.Embedding(vocab_struct, MOCK_HIDDEN)
        self.seq_head = nn.Linear(MOCK_HIDDEN, vocab_seq)
        self.struct_head = nn.Linear(MOCK_HIDDEN, vocab_struct)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> _MockOutput:
        hidden = (
            self.seq_embed(sequence_tokens.clamp(0, SEQ_VOCAB_SIZE - 1))
            + self.struct_embed(structure_tokens.clamp(0, STRUCT_VOCAB_SIZE - 1))
        )
        return _MockOutput(self.seq_head(hidden), self.struct_head(hidden))

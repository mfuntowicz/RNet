import torch
from torch.nn import Sequential, Module, ELU, Linear

from models import masked_softmax
from models.dropout import VariationalDropout


class DotAttention(Module):
    def __init__(self, in_size: int, attention_size: int, dropout_p: float,
                 slope: float = 0.01, mask_diag: bool = False):
        super().__init__()

        self._mask_diag = mask_diag

        self.input = Sequential(
            VariationalDropout(dropout_p),
            Linear(in_size, attention_size, bias=False),
            ELU(slope, True)
        )

        self.memory = Sequential(
            VariationalDropout(dropout_p),
            Linear(in_size, attention_size, bias=False),
            ELU(slope, True)
        )

    def forward(self, context, query, query_mask):
        input_, memory_ = self.input(context), self.memory(query)

        logits = torch.bmm(input_, memory_.transpose(2, 1)) / (input_.size(-1) ** 0.5)

        # In case of dot product between the two same element, avoid matching the element itself
        if self._mask_diag:
            idx = torch.arange(logits.size(-1), device=logits.device, dtype=torch.long)
            logits[:, idx, idx] = -torch.finfo(logits.dtype).max

        logits, score = masked_softmax(logits, query_mask)

        return torch.bmm(score, query)
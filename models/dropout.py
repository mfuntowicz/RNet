from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout


def variational_dropout(x: Tensor, p: float, training: bool) -> Tensor:
    mask = dropout(x.new_ones((1, ) + x.size()[1:]), p, training)

    return x * mask.expand_as(x)


class VariationalDropout(Module):
    def __init__(self, dropout_p):
        super().__init__()

        self._dropout_p = min(1, max(0, dropout_p))

    def forward(self, x):
        return variational_dropout(x, self._dropout_p, self.training)

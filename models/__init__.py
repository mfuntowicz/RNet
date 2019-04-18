from typing import Optional

import torch
from torch import finfo, Tensor
from torch.nn.functional import softmax


def mask_from_sequence_lengths(lengths: Tensor, max_length: Optional[int] = None) -> Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.
    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = lengths.new_ones((lengths.size(0), lengths.max() if not max_length else max_length), dtype=torch.uint8)
    range_tensor = ones.cumsum(dim=1)
    return lengths.unsqueeze(1) >= range_tensor


def masked_softmax(x, mask, dim=-1):
    """
    Perfoms a softmax over elements which are not padded.
    :param x: The padded sequence.
    :param mask: Binary Tensor filled with 1 for true data and 0 for padding.
    :param dim: The over which the softmax will be applied.
    :return:
    """
    INF = finfo(x.dtype).max

    if len(x.size()) == 3:
        mask = mask.unsqueeze(1)

    mask = mask.type_as(x)

    # To limit numerical errors from large vector elements outside the mask, we zero these out.
    x_ = x.masked_fill((1 - mask).byte(), -INF)
    return x_, softmax(x_, dim)


def normalize_predictions(p_start, p_end, max_len=15):
    preds_s, preds_e = [], []

    for i in range(p_start.size(0)):
        p_s, p_e = normalize_prediction(p_start[i], p_end[i], max_len)
        preds_s.append(p_s), preds_e.append(p_e)

    return preds_s, preds_e


def normalize_prediction(p_start, p_end, max_len=15):
    with torch.no_grad():
        """Take argmax of constrained score_s * score_e.
        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        import numpy as np

        # Outer product of scores to get full p_s * p_e matrix
        scores = torch.ger(p_start, p_end)

        # Zero out negative length and over-length span scores
        scores.triu_().tril_(max_len - 1)

        # Take argmax or top n
        scores_flat = scores.flatten()
        idx_sort = [scores_flat.argmax().item()]
        return tuple(idx[0] for idx in np.unravel_index(idx_sort, scores.shape))





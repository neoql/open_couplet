import torch

from typing import Optional


def sentence_pattern(sent: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
    """
    If sentence is
        [[1, 2, 3, 2, 4, 5, 2],
         [1, 2, 3, 4, 5, 0, 0]]
    pattern will be
        [[[1, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 1],
          [0, 0, 1, 0, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 1],
          [0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 1, 0, 0, 1]],

         [[1, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 1, 1],
          [0, 0, 0, 0, 0, 1, 1]]]
    :param sent: (batch, max_len) sentence
    :param pad_mask: (batch, max_len) the boolean mask
    :return: (batch, max_len, max_len) pattern
    """
    max_len = sent.size(1)
    pattern = torch.stack([sent] * max_len, dim=1) == sent.view(-1, max_len, 1)

    if pad_mask is not None:
        pattern = pattern.masked_fill(pad_mask.unsqueeze(1), 0)

    return pattern

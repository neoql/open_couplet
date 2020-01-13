import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        d = q.size(-1)

        # (*, q_len, kv_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, -1)
        return torch.matmul(weights, v), weights

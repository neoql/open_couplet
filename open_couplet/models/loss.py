import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    __constants__ = ['n_classes', 'epsilon', 'ignore_index', 'reduction']

    def __init__(self,
                 n_classes: int,
                 epsilon: float = 0.0,
                 ignore_index: int = -1,
                 reduction: str = 'mean'):
        super(LabelSmoothingLoss, self).__init__()

        assert reduction in ('sum', 'mean')

        self.n_classes = n_classes
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction

        less_val = 1 if self.ignore_index < 0 else 2
        one_hot = torch.full((n_classes,), self.epsilon / (n_classes - less_val))
        if ignore_index >= 0:
            one_hot[ignore_index] = 0.0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, inputs: torch.Tensor, target: torch.Tensor):
        """
        :param inputs: (batch_size, n_classes) log-probabilities
        :param target: (batch_size,)
        :return: scalar
        """
        assert self.n_classes == inputs.size(-1)

        ignore_indices = (target == self.ignore_index)

        dist = self.one_hot.repeat(target.size(0), 1)\
            .scatter_(-1, target.unsqueeze(-1), 1 - self.epsilon)\
            .masked_fill_(ignore_indices.unsqueeze(-1), 0.0)

        loss = self.criterion(inputs, dist)

        if self.reduction == 'mean':
            num = inputs.size(0) - ignore_indices.sum().item()
            loss = loss/num

        return loss


class Perplexity(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super(Perplexity, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, inputs: torch.Tensor, target: torch.Tensor):
        return torch.exp(self.criterion(inputs, target))

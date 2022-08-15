import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassBCEWithLogitsLoss(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    """

    def __init__(self, smoothing=0.0):
        super().__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        batch_size = x.size(0)
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (batch_size, num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        return F.binary_cross_entropy_with_logits(x, target, reduction='sum') / batch_size


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class MixupLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, inputs, targets_a, targets_b, lam):
        return lam * self.loss_fn(inputs, targets_a) + (1 - lam) * self.loss_fn(inputs, targets_b)

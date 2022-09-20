"""
Loss for training
"""
# https://github.com/kevinzakka/pytorch-goodies

import torch
import torch.nn.functional as F
from torch import nn


class JaccardLoss(nn.Module):
    """Computes the Jaccard loss, a.k.a the IoU loss"""

    def __init__(self):
        """Creates a criterion to measure the Jaccard loss, a.k.a the IoU loss."""
        super(JaccardLoss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        num_classes = logits.shape[1]
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return 1 - jacc_loss


def f1_score(pred, label):
    """
    Args:
        pred  : (N, H, W)
        label : (N, H, W)

    Returns:
        f1: float Tensor
    """
    pred = pred.squeeze().ravel()
    label = label.squeeze().ravel()

    assert pred.shape == label.shape
    tp = (label * pred).sum().to(torch.float32)
    tn = ((1 - label) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - label) * pred).sum().to(torch.float32)
    fn = (label * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = False
    return f1

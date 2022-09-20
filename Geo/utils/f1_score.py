# https://github.com/SuperShinyEyes
"""
Calculate function of F1 Score
"""
import torch


def f1_score(pred: torch.Tensor,
             label: torch.Tensor,
             is_training=False):
    """Calculate the f1-score of the images
    Binary Input required

    Keyword arguments:
    pred -- torch Tensor of Output of the model
    label -- torch Tensor of the label
    """
    pred = pred.squeeze().ravel()
    label = label.squeeze().ravel()

    assert pred.shape == label.shape
    tp = (label * pred).sum().to(torch.float32)
    tn = ((1 - label) * (1 - pred)).sum().to(torch.float32)  # not used
    fp = ((1 - label) * pred).sum().to(torch.float32)
    fn = (label * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

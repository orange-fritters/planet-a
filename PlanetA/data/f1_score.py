import torch


def f1_score(pred, label, is_training=False):
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
    f1.requires_grad = is_training
    return f1

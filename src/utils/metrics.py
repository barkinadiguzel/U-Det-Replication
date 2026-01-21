import torch

def dice_score(pred, target, epsilon=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

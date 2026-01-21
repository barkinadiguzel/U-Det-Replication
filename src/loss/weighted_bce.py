import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        loss = - (self.pos_weight * target * torch.log(pred + 1e-8) +
                  (1 - target) * torch.log(1 - pred + 1e-8))
        return loss.mean()

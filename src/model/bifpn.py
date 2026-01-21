import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPNBlock(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for ch in channels_list])
        self.bns = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels_list])
        self.relu = nn.ReLU()

    def forward(self, features):
        fused = []
        for i, f in enumerate(features):
            x = self.convs[i](f)
            x = self.bns[i](x)
            x = self.relu(x)
            fused.append(x)
        return fused

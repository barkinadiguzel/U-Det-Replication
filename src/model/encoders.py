import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Mish(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Mish()
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(base_ch*4, base_ch*8, dropout=True)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlock(base_ch*8, base_ch*16)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(self.pool1(f1))
        f3 = self.conv3(self.pool2(f2))
        f4 = self.conv4(self.pool3(f3))
        f5 = self.conv5(self.pool4(f4))
        return [f1, f2, f3, f4, f5]

import torch
import torch.nn as nn
from encoders import ConvBlock

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.up4 = UpConv(base_ch*16, base_ch*8)
        self.up3 = UpConv(base_ch*8, base_ch*4)
        self.up2 = UpConv(base_ch*4, base_ch*2)
        self.up1 = UpConv(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        x = self.up4(f5, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        return torch.sigmoid(self.final(x))

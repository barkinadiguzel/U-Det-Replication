import torch
import torch.nn as nn
from encoders import Encoder
from bifpn import BiFPNBlock
from decoder import Decoder

class UDet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch)
        self.bifpn = BiFPNBlock([base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16])
        self.decoder = Decoder(base_ch)

    def forward(self, x):
        enc_feats = self.encoder(x)
        fused_feats = self.bifpn(enc_feats)
        out = self.decoder(fused_feats)
        return out

from os.path import join as pjoin
import torch
import torch.nn as nn
from typing import Sequence
from .block import MSAFB, DWConv2d, DWConv2dReLU, CA_Block, CBAMBlock, MSWTB

class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    )-> None:
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv1 = DWConv2dReLU(
            out_channels * 2,
            out_channels,
        )
        self.conv2 = DWConv2dReLU(
            out_channels,
            out_channels//2,
        )

        self.conv3 = DWConv2dReLU(
            out_channels//2,
            out_channels,
        )
        self.mswtb2 = nn.Sequential(
            MSWTB(out_channels),
            MSWTB(out_channels),
        )

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # upsample
        out = self.up(input)
        # cat
        out = torch.cat([out, skip], dim=1)
        # dwcon2d
        out = self.conv1(out)
        #  resnetblock
        identify = out
        out = self.conv2(out)
        out = self.conv3(out)
        out = identify + out
        # mswtb * 2
        out = self.mswtb2(out)
        return out
    
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=9, scale_factor=2):
        super().__init__()
        self.final_up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv_final = DWConv2d(in_channels, num_classes)

    def forward(self, x):
        return self.conv_final(self.final_up(x))


class ACMUNetDecoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_classes: int,
    )-> None:
        super(ACMUNetDecoder, self).__init__()

        self.layers = nn.ModuleList()
        
        for i in range(1, len(dims)):
            self.layers.append(UpBlock(dims[i-1], dims[i]))

        self.head = SegmentationHead(dims[-1], num_classes)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        out = features[0]
        features = features[1:]
        for i, layer in enumerate(self.layers):
            out = layer(out, features[i])
        return self.head(out)

    

import torch.nn as nn
from .encoder import ACMUnetEncoder
from .decoder import ACMUNetDecoder
import torch


class ACMUnet(nn.Module):
    def __init__(self, in_channels, num_classes=9):
        super().__init__()
        self.encoder = ACMUnetEncoder(in_channels=in_channels)
        self.dims = self.encoder.dims
        self.decoder = ACMUNetDecoder(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, input):
        results = self.encoder(input)
        return self.decoder(results[::-1])
    
    @torch.no_grad()
    def freeze_encoder(self) -> None:
        pass

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        pass
    

def build_model(**kwargs):
    in_channels = kwargs.get("in_channels", 3)
    num_classes = kwargs.get("num_classes", 9)
    return ACMUnet(in_channels=in_channels, num_classes=num_classes)
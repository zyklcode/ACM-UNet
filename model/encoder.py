from torchinfo import summary
from torchvision import models
import torch
import torch.nn as nn
import os
from .vmamba import ENCODERS
from .block import DWConv2dReLU

PARENT_PATH = os.path.dirname(os.path.abspath(__file__))



class ACMUnetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ACMUnetEncoder, self).__init__()
        resnet = models.resnet50()
        resnet.load_state_dict(torch.load(os.path.join(PARENT_PATH, "pretrain", "resnet50-19c8e357.pth")))
        vmamba = ENCODERS["tiny_0230s"](in_channels=in_channels)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        ) # (B, 64, 112, 112)

        self.layer1 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1,
        ) # (B, 256, 56, 56)
        self.dims = [64, 256, 512, 1024]

        self.layer2 = resnet.layer2 # (B, 512, 28, 28)

        self.layer3 = resnet.layer3 # (B, 1024, 14, 14)


        self.adapter1 = nn.Sequential(
            nn.Conv2d(1024, 384, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.adapter2 = nn.Sequential(
            nn.Conv2d(384, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.layer_vmamba = vmamba.layers[2].blocks[0:2]

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        results = []
        x = self.layer0(x)  # (B, 64, 112, 112)
        results.append(x)
        x = self.layer1(x)  # (B, 256, 56, 56)
        results.append(x)
        x = self.layer2(x)  # (B, 512, 28, 28)
        results.append(x)
        x = self.layer3(x)  # (B, 1024, 14, 14)

        x = self.adapter1(x) # (B, 384, 14, 14)

        x = self.layer_vmamba(x)
        
        x = self.adapter2(x)

        results.append(x)

        return results

# <Project>/models/resnet_enhancer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNetEnhancer(nn.Module):
    """
    ResNet18 encoder + simple decoder for image enhancement.
    Input:  (B,3,H,W) in [0,1]
    Output: (B,3,H,W) in [0,1]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet18(weights="DEFAULT" if pretrained else None)

        # take layers up to layer4
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # /4
        self.layer2 = backbone.layer2  # /8
        self.layer3 = backbone.layer3  # /16
        self.layer4 = backbone.layer4  # /32

        # decoder: progressively upsample + conv
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256, 128)
        self.up2 = self._up_block(128, 64)
        self.up1 = self._up_block(64, 32)

        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def _up_block(self, cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # encoder
        x0 = self.stem(x)      # /4, 64ch
        x1 = self.layer1(x0)  # /4, 64ch
        x2 = self.layer2(x1)  # /8, 128ch
        x3 = self.layer3(x2)  # /16, 256ch
        x4 = self.layer4(x3)  # /32, 512ch

        # decoder
        d4 = self.up4(x4)  # /16
        d3 = self.up3(d4)  # /8
        d2 = self.up2(d3)  # /4
        d1 = self.up1(d2)  # /2
        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)  # back to H,W

        out = torch.sigmoid(self.out_conv(d0))
        return out

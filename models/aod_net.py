# <Project>/models/aod_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AODNet(nn.Module):
    """
    AOD-Net style K-estimation:
      K = f(I)
      J = K*I - K + b  (b implicit via last conv bias)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        x3 = F.relu(self.conv3(torch.cat([x1, x2], 1)))
        x4 = F.relu(self.conv4(torch.cat([x2, x3], 1)))

        k = F.relu(self.conv5(torch.cat([x1, x2, x3, x4], 1)))

        # enhanced
        j = k * x - k + 1.0  # +1 helps avoid dark collapse
        j = torch.clamp(j, 0.0, 1.0)
        return j, k

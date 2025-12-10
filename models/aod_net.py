# <Project>/models/aod_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AODNet(nn.Module):
    """
    AOD-Net style K-estimation:
      K = f(I)
      J = K*I - K + 1  (so if K ≡ 1, J = I)
    Identity-style init:
      - conv1..conv4: weights=0, bias=0
      - conv5:        weights=0, bias=1
      => at init: K ≡ 1, J ≡ I
    """
    def __init__(self, identity_init: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1)

        if identity_init:
            self._init_identity()

    def _init_identity(self):
        """
        Initialize so that at start:
          x1 = x2 = x3 = x4 = 0
          k  = 1
          j  = x
        """
        # conv1..conv4: zero weights & zero bias => outputs = 0
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        # conv5: zero weights, bias = 1 => output = 1 (per-channel constant)
        nn.init.zeros_(self.conv5.weight)
        if self.conv5.bias is not None:
            self.conv5.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        x3 = F.relu(self.conv3(torch.cat([x1, x2], 1)))
        x4 = F.relu(self.conv4(torch.cat([x2, x3], 1)))

        k = F.relu(self.conv5(torch.cat([x1, x2, x3, x4], 1)))

        # enhanced
        j = k * x - k + 1.0  # if k == 1, j == x
        j = torch.clamp(j, 0.0, 1.0)
        return j, k
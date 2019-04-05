#!/bin/python3

#!/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, dilation, bn=False):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels
        self.bn = bn
        self.conv1 = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2)
        if self.bn:
            self.bn1 = nn.BatchNorm1d(num_channels)
            self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        if self.bn:
            in_ = self.bn1(x)
        else:
            in_ = x

        a = self.conv1(F.relu(in_))

        if self.bn:
            a = self.bn2(a)

        b = self.conv2(F.relu(a))

        return x + b


class Path(nn.Module):
    def __init__(self,
                 in_channels=6,
                 num_channels=[32, 32, 64, 128],
                 kernel_size=3,
                 dilations=[1, 1, 1, 1],
                 num_components=1,
                 bn=False):
        super(Path, self).__init__()
        num_blocks = len(num_channels) - 1
        if len(dilations) != num_blocks:
            msg = ("Number of dilations must be equal to number of residual "
                   "blocks.")
            raise ValueError(msg)

        self.bn = bn
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv1d(
            in_channels,
            num_channels[0],
            kernel_size=kernel_size,
            dilation=1,
            padding=(kernel_size - 1) // 2)

        self.blocks = nn.ModuleList([
            ResidualBlock(
                num_channels[i],
                kernel_size=kernel_size,
                dilation=dilations[i],
                bn=self.bn) for i in range(self.num_blocks)
        ])

        self.convs = nn.ModuleList([
            nn.Conv1d(
                num_channels[i],
                num_channels[i+1],
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2)
            for i in range(self.num_blocks)
        ])

        if self.bn:
            self.bn_out = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        tap = self.conv1(x)

        for i in range(self.num_blocks):
            tap = F.avg_pool1d(self.blocks[i](tap), 2, 2)
            #tap = self.blocks[i](tap)
            tap = F.relu(self.convs[i](tap))

        return tap


class Discriminator(nn.Module):
    def __init__(self, in_channels=2, num_channels=[32, 32, 32, 32, 32]):
        super(Discriminator, self).__init__()
        self.a = Path(in_channels=in_channels, num_channels=num_channels)
        self.b = Path(in_channels=in_channels, num_channels=num_channels)
        self.c = Path(in_channels=in_channels, num_channels=num_channels)

        self.fc = nn.Linear(num_channels[-1] * 3 * 40, 1)
        for m in self.modules():
            if 'weight' in m._parameters:
                nn.utils.spectral_norm(m)

    def forward(self, x):
        a = x[:, 0, ...]
        b = x[:, 1, ...]
        c = x[:, 2, ...]

        #feats = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1).reshape(a.shape[0], -1)#.mean(dim=2)
        feats = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1).reshape(a.shape[0], -1)#.mean(dim=2)

        logits = self.fc(feats)

        return logits


if __name__ == '__main__':
    disc = Discriminator()
    disc(torch.randn(32, 3, 2, 160))

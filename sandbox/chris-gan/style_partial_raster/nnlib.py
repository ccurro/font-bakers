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
                 num_channels=32,
                 num_blocks=4,
                 kernel_size=3,
                 dilations=[1, 1, 1, 1],
                 num_components=1,
                 bn=False):
        super(Path, self).__init__()

        if len(dilations) != num_blocks:
            msg = ("Number of dilations must be equal to number of residual "
                   "blocks.")
            raise ValueError(msg)

        self.bn = bn
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv1d(
            in_channels,
            num_channels,
            kernel_size=kernel_size,
            dilation=1,
            padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResidualBlock(
                num_channels,
                kernel_size=kernel_size,
                dilation=dilations[i],
                bn=self.bn) for i in range(self.num_blocks)
        ])

        if self.bn:
            self.bn_out = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        taps = [self.conv1(x)]

        for i in range(self.num_blocks):
            tap = self.blocks[i](taps[i])
            taps.append(tap)

        z = self.conv2(taps[-1])

        if self.bn:
            z = self.bn_out(z)

        return F.relu(z)

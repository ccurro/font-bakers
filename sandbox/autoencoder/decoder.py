#!/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MiniBlock(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(MiniBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            num_channels, num_channels, kernel_size, bias=False, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(
            num_channels, num_channels, kernel_size, bias=False, padding=(kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        a = F.relu(self.bn1(x))
        a = self.conv1(a)

        a = F.relu(self.bn2(a))
        a = self.conv2(a)

        return a + x


class Block(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(Block, self).__init__()

        self.mini_block1 = MiniBlock(num_channels, kernel_size)
        self.mini_block2 = MiniBlock(num_channels, kernel_size)

    def forward(self, x):
        a = self.mini_block1(x)
        a = self.mini_block2(a)

        return a


class Decoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_channels=[64, 64, 64, 64],
                 kernel_size=3
                 ):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, num_channels[0], kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.transitions = nn.ModuleDict(
            {(str(i),
              nn.Conv1d(num_channels[i], num_channels[i+1], kernel_size=1))
             for i, a in enumerate(np.diff(num_channels)) if a > 0})

        self.blocks = nn.ModuleList(
            [Block(c, kernel_size) for c in num_channels])

        self.conv_out = nn.Conv1d(
            num_channels[-1], 3*4, kernel_size=1)

        self.called = False

    def forward(self, x):
        a = self.conv1(x)

        for i, block in enumerate(self.blocks):
            a = block(a)
            if str(i) in self.transitions:
                transition = self.transitions[str(i)]
                a = transition(a)

        if not self.called:
            print(a.shape)
            self.called = True

        a = self.conv_out(a).reshape(a.shape[0], 3, 4, -1)
                    
        return torch.cat([torch.roll(a[:, :, 2:, :], 1, dims=-1), a], dim=2)


if __name__ == "__main__":
    dec = Decoder()
    z = dec(torch.randn(16, 2, 256))
    print(z.shape)

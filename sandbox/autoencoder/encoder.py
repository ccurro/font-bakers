#!/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MiniBlock(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(MiniBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size, bias=False, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size, bias=False, padding=(kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

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


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_channels=[16, 32, 32, 64, 64, 128, 256],
                 kernel_size=3,
                 pooling=None):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, num_channels[0], kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.transitions = nn.ModuleDict(
            {(str(i),
              nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=1))
             for i, a in enumerate(np.diff(num_channels)) if a > 0})
        print(self.transitions)

        self.blocks = nn.ModuleList(
            [Block(c, kernel_size) for c in num_channels])

        self.pooling = pooling

        self.fc = nn.Linear(256*2*2, 32)

        self.called = False

    def forward(self, x):
        a = self.conv1(x)

        for i, block in enumerate(self.blocks):
            a = block(a)
            if str(i) in self.transitions:
                transition = self.transitions[str(i)]
                a = transition(a)

            if self.pooling is not None:
                a = self.pooling(a)

        if not self.called:
            print(a.shape)
            self.called = True
            
        return self.fc(a.reshape(a.shape[0], -1))


if __name__ == "__main__":
    enc = Encoder(pooling=nn.MaxPool2d(2,2))
    z = enc(torch.randn(32, 2, 256, 256))
    print(z.shape)

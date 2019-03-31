#!/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F


def AdaIN(x, gains, biases):
    assert len(x.shape) == 3
    eps = 1e-8
    mean = x.mean(2, keepdim=True)
    rstddev = torch.rsqrt(x.var(2, keepdim=True) + eps)

    normed = (x - mean) * rstddev

    return gains * normed + biases


class StyleBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 style_dim,
                 kernel_size=3,
                 dilation=1,
                 init_width=None,
                 initial=False):
        super(StyleBlock, self).__init__()
        self.num_channels = num_channels
        self.init_width = init_width
        if initial:
            self.initial = nn.parameter.Parameter(
                torch.randn(1, num_channels, init_width), requires_grad=True)

        self.noise_gain = nn.parameter.Parameter(
            torch.zeros(1, num_channels, 1), requires_grad=True)

        self.conv1 = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2)

        self.fc = nn.Linear(style_dim, 2 * num_channels)

    def forward(self, style, x=None):
        if x is None:
            a = self.conv1(self.initial)
        else:
            a = self.conv1(x)

        gains_and_biases = self.fc(style)
        gains = gains_and_biases[:, :self.num_channels].unsqueeze(-1)
        biases = gains_and_biases[:, self.num_channels:].unsqueeze(-1)

        noise = self.noise_gain * torch.randn(self.noise_gain.shape)

        return AdaIN(F.relu(a + noise), gains, biases)


class MappingNet(nn.Module):
    def __init__(self, z_dim, width, num_layers):
        super(MappingNet, self).__init__()

        self.fc = nn.Linear(z_dim, width)

        self.layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(num_layers)])

    def forward(self, z):
        a = F.elu(self.fc(z))

        for layer in self.layers:
            a = F.elu(a)

        return a


class StyleNet(nn.Module):
    def __init__(self, num_curves, num_blocks, style_dim, z_dim, num_channels):
        super(StyleNet, self).__init__()

        init_width = int(num_curves / 2**(num_blocks / 2))
        print("init_width:", init_width)

        self.mapping = MappingNet(z_dim, style_dim, 4)
        self.styleblock1 = StyleBlock(
            num_channels,
            style_dim=style_dim,
            init_width=init_width,
            initial=True)
        self.styleblock2 = StyleBlock(num_channels, style_dim=style_dim)
        self.styleblocks = nn.ModuleList([
            StyleBlock(num_channels, style_dim=style_dim)
            for _ in range(num_blocks)
        ])

        self.out = nn.Conv1d(num_channels, 4, kernel_size=1)

    def forward(self, z):
        style = self.mapping(z)
        a = self.styleblock1(style)
        a = self.styleblock2(style, x=a)

        styleblocks = iter(self.styleblocks)

        for block in styleblocks:
            a = F.interpolate(a, scale_factor=2)
            a = block(style, x=a)
            a = next(styleblocks)(style, x=a)

        implicits = self.out(a)

        return torch.cat(
            [torch.roll(implicits[:, 2:, :], 1, dims=-1), implicits], dim=1)


if __name__ == '__main__':
    z_dim = 16
    style_dim = 128

    style_net = StyleNet(40, 4, style_dim, z_dim, 32)
    style_net(torch.randn(16, z_dim))

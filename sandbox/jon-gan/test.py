import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torch.utils import data

from nnlib import Path
from util import raster, lissajous_walk
from FixedSizeFontData import Dataset

from style import StyleNet

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

batch_size = 512
z_dim = 16
num_curves = 32  # pow 2 ish
style_dim = 128
num_blocks = 8
num_channels = 32

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, num_channels=32):
        super(Discriminator, self).__init__()
        num_blocks = 8
        dilations = [1 for _ in range(num_blocks)]
        self.a = Path(in_channels=in_channels, num_channels=num_channels)
        self.b = Path(in_channels=in_channels, num_channels=num_channels)
        self.c = Path(in_channels=in_channels, num_channels=num_channels)

        self.fc = nn.Linear(num_channels * 3, 1)
        for m in self.modules():
            if 'weight' in m._parameters:
                nn.utils.spectral_norm(m)

    def forward(self, x):
        a = x[:, 0, ...]
        b = x[:, 1, ...]
        c = x[:, 2, ...]

        #feats = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1).reshape(a.shape[0], -1)#.mean(dim=2)
        feats = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1).mean(dim=2)
        logits = self.fc(feats)

        return logits


class Generator(nn.Module):
    def __init__(self, num_curves, num_blocks, style_dim, z_dim, num_channels):
        super(Generator, self).__init__()

        self.style_net1 = StyleNet(num_curves, num_blocks, style_dim, z_dim,
                                   num_channels)
        self.style_net2 = StyleNet(num_curves, num_blocks, style_dim, z_dim,
                                   num_channels)
        self.style_net3 = StyleNet(num_curves, num_blocks, style_dim, z_dim,
                                   num_channels)

    def forward(self, z):
        return self.style_net1(z), self.style_net2(z), self.style_net3(z)




gen = Generator(num_curves, num_blocks, style_dim, z_dim, num_channels)
checkpoint = torch.load("outs15/checkpoint0000012000.pt")
gen.load_state_dict(checkpoint['gen'])

gen.eval()

for i in range(20):
    z = torch.rand(1, z_dim)
    a, b, c = gen(z)
    a = a.cpu().detach().numpy().transpose(0, 2, 1).reshape(-1, 3, 2).transpose(0, 2, 1).astype(np.float64)
    b = b.cpu().detach().numpy().transpose(0, 2, 1).reshape(-1, 3, 2).transpose(0, 2, 1).astype(np.float64)
    c = c.cpu().detach().numpy().transpose(0, 2, 1).reshape(-1, 3, 2).transpose(0, 2, 1).astype(np.float64)
    raster([[a, b, c]])
    plt.savefig("test{}.png".format(i))
    plt.close()


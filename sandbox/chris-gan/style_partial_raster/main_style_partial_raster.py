#!/bin/python

from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torch.utils import data

from nnlib import Path
from data import FontData
from util import raster

from style import StyleNet

import matplotlib.pyplot as plt

from itertools import chain


def sample_qb(control_points, steps):
    """
    cp -> [batch, 6, num_curves]
    """
    batch_size = control_points.shape[0]
    num_curves = control_points.shape[2]
    cp = control_points.permute(0, 2, 1).reshape(-1, num_curves, 3, 2)
    steps = torch.linspace(0, 1, steps).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    P0 = cp[..., 0, :].unsqueeze(-1)
    P1 = cp[..., 1, :].unsqueeze(-1)
    P2 = cp[..., 2, :].unsqueeze(-1)

    samples = (
        P1 + (1 - steps)**2 * (P0 - P1) + steps**2 * (P2 - P1)).transpose(
            -1, -2)
    samples = samples.reshape(batch_size, -1, 2).transpose(-1, -2)

    return samples


class Discriminator(nn.Module):
    def __init__(self, in_channels=2, num_channels=32):
        super(Discriminator, self).__init__()
        self.a = Path(in_channels=in_channels, num_channels=num_channels)
        self.b = Path(in_channels=in_channels, num_channels=num_channels)
        self.c = Path(in_channels=in_channels, num_channels=num_channels)

        self.fc = nn.Linear(num_channels * 2, 1)
        for m in self.modules():
            if 'weight' in m._parameters:
                nn.utils.spectral_norm(m)

    def forward(self, a, b, c):
        a = sample_qb(a, 10)
        b = sample_qb(b, 10)
        c = sample_qb(c, 10)
        #feats = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1).reshape(a.shape[0], -1)#.mean(dim=2)
        feats = torch.cat([self.a(a), self.b(b)], dim=1).mean(dim=2)
        logits = self.fc(feats)

        return logits


def pad(in_, like):
    out = F.pad(in_, [
        like.shape[2] - in_.shape[2], 0, 0, 0, 0, like.shape[0] - in_.shape[0]
    ])
    return out


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


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 1024
    z_dim = 16
    num_curves = 64 # pow 2 ish
    style_dim = 128
    num_blocks = 8
    num_channels = 32

    z_test = torch.randn(1, z_dim)

    gen = Generator(num_curves, num_blocks, style_dim, z_dim, num_channels)

    disc = Discriminator()
    #fd = FontData(max_num_examples=50000, max_num_pts_per_contour=num_curves)
    ds = Dataset(batch_size, max_num_curves=num_curves)
    dl = data.DataLoader(ds, num_workers=16, pin_memory=True)

    optim_gen = optim.Adam(gen.parameters(), 2e-4, [0.5, 0.9])
    optim_disc = optim.Adam(disc.parameters(), 2e-4, [0.5, 0.9])

    tic = time()
    for i, real_data in enumerate(dl):
        a = torch.tensor(real_data[0]).squeeze().cuda()
        b = torch.tensor(real_data[1]).squeeze().cuda()
        c = torch.tensor(real_data[2]).squeeze().cuda()

        if i % 5 == 0:
            for _ in range(1):
                z = torch.randn(a.shape[0], z_dim)
                optim_gen.zero_grad()
                fake_data = gen(z)
                d_fake = disc(*fake_data)
                gen_loss = d_fake.mean()
                gen_loss.backward(-torch.ones_like(gen_loss))
                optim_gen.step()

        optim_disc.zero_grad()

        with torch.no_grad():
            gen.eval()
            z = torch.randn(a.shape[0], z_dim)
            fake_data = gen(z)
            gen.train()

        d_real = disc(a, pad(b, a), pad(c, a))
        d_fake = disc(fake_data[0], fake_data[1], fake_data[2])

        disc_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()

        if disc_loss > 0:
            disc_loss.backward(torch.ones(1))
            optim_disc.step()

        print(i,
              disc_loss.cpu().detach().numpy(),
              "{0:.2f}s".format(time() - tic))
        tic = time()

        if i % 100 == 0:
            gen.eval()
            for j in range(1):
                a, b, c = gen.forward(z_test)
                a = a.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                b = b.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                #c = c.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                #    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)

                raster([[a, b]])  #, c]])
                plt.savefig("outs_partial_raster/test{:010d}.png".format(i))
                plt.close()
            gen.train()

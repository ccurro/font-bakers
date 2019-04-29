#!/bin/python

import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torch.utils import data

from nnlib import Path
from util import raster, lissajous_walk
from data import Dataset, CHARACTER_INDICES

from style import StyleNet

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from itertools import chain
from tqdm import tqdm


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
        self.d = Path(in_channels=3 * num_channels, num_channels=num_channels)
        self.fc = nn.Linear(num_channels, 1)

        for m in self.modules():
            if 'weight' in m._parameters:
                nn.utils.spectral_norm(m)

    def forward(self, x):
        a = x[:, 0, ...]
        b = x[:, 1, ...]
        c = x[:, 2, ...]

        d = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1)
        feats = self.d(d).mean(dim=2)
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

    def forward(self, z, classes):
        return self.style_net1(z, classes), self.style_net2(
            z, classes), self.style_net3(z, classes)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 512
    z_dim = 16
    num_curves = 32  # pow 2 ish
    style_dim = 128
    num_blocks = 8
    num_channels = 32

    lissajous = lissajous_walk(z_dim)

    gen = Generator(num_curves, num_blocks, style_dim, z_dim, num_channels)

    disc = Discriminator()

    num_pts = int(160 / num_curves)
    ds = Dataset("/zooper1/fontbakers/data/renders/160_samples_3/")
    dl = data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True)

    mapping_params = [
        param for name, param in gen.named_parameters() if "mapping" in name
    ]
    other_params = [
        param for name, param in gen.named_parameters()
        if "mapping" not in name
    ]
    optim_gen = optim.Adam([{
        'params': mapping_params,
        'lr': 2e-5
    }, {
        'params': other_params
    }], 2e-4, [0.5, 0.9])
    optim_disc = optim.Adam(disc.parameters(), 2e-4, [0.5, 0.9])

    tic = time()
    for i, (real_data, classes) in enumerate(iter(cycle(dl))):
        real_data = real_data.cuda()
        classes = classes[0].cuda()

        if i % 2 == 0:
            for _ in range(1):
                z = torch.rand(real_data.shape[0], z_dim)
                optim_gen.zero_grad()
                a, b, c = gen(z, classes)
                fake_data = torch.stack([
                    sample_qb(a, num_pts),
                    sample_qb(b, num_pts),
                    sample_qb(c, num_pts)
                ],
                                        dim=1)
                d_fake = disc(fake_data)
                gen_loss = d_fake.mean()
                gen_loss.backward(-torch.ones_like(gen_loss))
                optim_gen.step()

        optim_disc.zero_grad()

        with torch.no_grad():
            gen.eval()
            z = torch.randn(real_data.shape[0], z_dim)
            classes = torch.randint(
                len(CHARACTER_INDICES), [real_data.shape[0]])
            a, b, c = gen(z, classes)
            fake_data = torch.stack([
                sample_qb(a, num_pts),
                sample_qb(b, num_pts),
                sample_qb(c, num_pts)
            ],
                                    dim=1)
            gen.train()

        d_real = disc(real_data)
        d_fake = disc(fake_data)

        disc_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()

        if disc_loss > 0:
            disc_loss.backward(torch.ones(1))
            nn.utils.clip_grad_norm_(disc.parameters(), 2)
            optim_disc.step()

        print(i,
              disc_loss.cpu().detach().numpy(),
              "{0:.2f}s".format(time() - tic))
        tic = time()

        if i % 5000 == 0:
            gen.eval()
            for j, z_test in tqdm(enumerate(lissajous)):
                a, b, c = gen(
                    z_test.unsqueeze(0),
                    torch.tensor([CHARACTER_INDICES['g']]))
                a = a.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                b = b.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                c = c.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)

                raster([[a, b, c]])
                plt.savefig("outs/test{:010d}_{:05d}.png".format(i, j))
                plt.close()
            cmd = (
                "ffmpeg -framerate 22 -pattern_type glob -i 'outs/test{:010d}*.png' "
                "-c:v libx264 -r 30 -pix_fmt yuv420p 'outs/out{:010d}.mp4' "
                "> /dev/null".format(i, i))
            os.system(cmd)
            os.system("rm outs/test{:010d}_*.png".format(i))
            gen.train()

            checkpoint = {
                "iter": i,
                "gen": gen,
                "disc": disc,
                "optim_gen": optim_gen,
                "optim_disc": optim_disc,
            }
            torch.save(checkpoint, "outs/checkpoint{:010d}.pt".format(i))

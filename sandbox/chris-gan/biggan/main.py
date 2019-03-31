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

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self,
                 num_curves,
                 latent_dim,
                 in_channels=16,
                 out_channels=6,
                 num_channels=32):
        super(Generator, self).__init__()
        self.dim = latent_dim
        self.project = nn.Linear(self.dim, in_channels * num_curves)
        self.bn = nn.BatchNorm1d(in_channels)

        self.a = Path(
            in_channels=in_channels, num_channels=num_channels, bn=True)
        self.b = Path(
            in_channels=in_channels, num_channels=num_channels, bn=True)
        self.c = Path(
            in_channels=in_channels, num_channels=num_channels, bn=True)

        self.a_out = Path(
            in_channels=num_channels, num_channels=num_channels, bn=True)
        self.b_out = Path(
            in_channels=num_channels, num_channels=num_channels, bn=True)
        self.c_out = Path(
            in_channels=num_channels, num_channels=num_channels, bn=True)

        self.first_contour = nn.Conv1d(
            num_channels, out_channels, kernel_size=1)
        self.second_contour = nn.Conv1d(
            num_channels, out_channels, kernel_size=1)
        self.third_contour = nn.Conv1d(
            num_channels, out_channels, kernel_size=1)

    def forward(self, shape, z=None):
        if z is None:
            z = torch.randn(shape[0], self.dim)

        z = F.relu(self.bn(self.project(z).reshape(shape)))

        a, b, c = self.a(z), self.b(z), self.c(z)

        a_mixed = a / 2 + (b + c) / 4
        b_mixed = b / 2 + (a + c) / 4
        c_mixed = c / 2 + (a + b) / 4

        a_out, b_out, c_out = self.a_out(a_mixed), self.b_out(
            b_mixed), self.c_out(c_mixed)

        first_contour = self.first_contour(a_out)
        second_contour = self.second_contour(b_out)
        third_contour = self.third_contour(c_out)

        return first_contour, second_contour, third_contour


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, num_channels=32):
        super(Discriminator, self).__init__()
        self.a = Path(in_channels=in_channels, num_channels=num_channels)
        self.b = Path(in_channels=in_channels, num_channels=num_channels)
        self.c = Path(in_channels=in_channels, num_channels=num_channels)

        self.fc = nn.Linear(num_channels * 2, 1)
        for m in self.modules():
            if 'weight' in m._parameters:
                nn.utils.spectral_norm(m)

    def forward(self, a, b, c):
        #feats = torch.cat([self.a(a), self.b(b), self.c(c)], dim=1).reshape(a.shape[0], -1)#.mean(dim=2)
        feats = torch.cat([self.a(a), self.b(b)], dim=1).mean(dim=2)
        logits = self.fc(feats)

        return logits


def pad(in_, like):
    out = F.pad(in_, [
        like.shape[2] - in_.shape[2], 0, 0, 0, 0, like.shape[0] - in_.shape[0]
    ])
    return out


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 2048
    latent_dim = 16
    num_curves = 60
    z_test = torch.randn(1, latent_dim)

    gen = Generator(num_curves, latent_dim)
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
                optim_gen.zero_grad()
                fake_data = gen.forward([batch_size, latent_dim, num_curves])
                d_fake = disc(*fake_data)
                gen_loss = d_fake.mean()
                gen_loss.backward(-torch.ones_like(gen_loss))
                optim_gen.step()

        optim_disc.zero_grad()

        with torch.no_grad():
            gen.eval()
            fake_data = gen.forward([batch_size, latent_dim, num_curves])
            gen.train()

        d_fake = disc(fake_data[0], fake_data[1], fake_data[2])

        d_real = disc(a, pad(b, a), pad(c, a))

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
                a, b, c = gen.forward([1, latent_dim, num_curves], z_test)
                a = a.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                b = b.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                #c = c.cpu().detach().numpy().transpose(0, 2, 1).reshape(
                #    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)

                raster([[a, b]])  #, c]])
                plt.savefig("outs/test{:010d}.png".format(i))
                plt.close()
            gen.train()

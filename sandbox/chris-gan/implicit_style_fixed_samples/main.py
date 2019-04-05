#!/bin/python

from time import time

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from torch.utils import data
from glob import glob

from util import raster
from walk import walk
from FixedSizeFontData import Dataset

from style import StyleNet, MappingNet
from discriminator import Discriminator

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import subprocess

from itertools import chain

from absl import flags, app
FLAGS = flags.FLAGS

flags.DEFINE_bool("test", False, "Infer or train")

def sample_qb(control_points, steps):
    """
    cp -> [batch, 6, num_curves]
    """
    batch_size = control_points.shape[0]
    num_curves = control_points.shape[2]
    cp = control_points.permute(0, 2, 1).reshape(-1, num_curves, 3, 2)
    steps = torch.linspace(0.05, .95, steps).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    P0 = cp[..., 0, :].unsqueeze(-1)
    P1 = cp[..., 1, :].unsqueeze(-1)
    P2 = cp[..., 2, :].unsqueeze(-1)

    samples = (
        P1 + (1 - steps)**2 * (P0 - P1) + steps**2 * (P2 - P1)).transpose(
            -1, -2)
    samples = samples.reshape(batch_size, -1, 2).transpose(-1, -2)

    return samples


class Generator(nn.Module):
    def __init__(self, num_curves, num_blocks, style_dim, z_dim, num_channels):
        super(Generator, self).__init__()

        self.style_net1 = StyleNet(num_curves, num_blocks, style_dim, z_dim,
                                   num_channels)
        self.style_net2 = StyleNet(num_curves, num_blocks, style_dim, z_dim,
                                   num_channels)
        self.style_net3 = StyleNet(num_curves, num_blocks, style_dim, z_dim,
                                   num_channels)

    def forward(self, z, mapping):
        return self.style_net1(z, mapping), self.style_net2(
            z, mapping), self.style_net3(z, mapping)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def save(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
                
    torch.save(model.state_dict(), path)

def load(gen, path):
    models = glob(path)
    model_file = max(models, key=os.path.getctime)
    gen.load_state_dict(torch.load(model_file))


def main(argv):
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("UTF-8")    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 256
    z_dim = 32
    num_curves = 64  # pow 2 ish
    style_dim = 32
    num_blocks = 8
    num_channels = 32
    num_pts = 10  #int(32*5 / num_curves)    

    z_test_c = torch.randn(1, z_dim)
    z_test = torch.tensor(walk(z_dim, num_samples=300))

    mapping = MappingNet(z_dim, style_dim, 2)
    gen = Generator(num_curves, num_blocks, style_dim, z_dim, num_channels)

    if FLAGS.test:
        load(gen, "saved_models/gen/{}/*".format(git_hash))
        load(mapping, "saved_models/mapping/{}/*".format(git_hash))        
        exit()

    disc = Discriminator()

    ds = Dataset("../data/with_640_samples/")
    dl = data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True)

    optim_gen = optim.Adam([{
        'params': mapping.parameters(),
        'lr': 1e-5
    }, {
        'params': gen.parameters()
    }], 1e-3, [0.5, 0.9])
    optim_disc = optim.Adam(disc.parameters(), 1e-3, [0.5, 0.9])

    tic = time()
    for i, real_data in enumerate(iter(cycle(dl))):
        real_data = real_data.cuda()
        if (i % 5 == 0) and (i > 0):
            #print(real_data.shape)
            #exit()
            for _ in range(1):
                z = torch.randn(real_data.shape[0], z_dim)
                optim_gen.zero_grad()
                a, b, c = gen(z, mapping)
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
            a, b, c = gen(z, mapping)
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
            optim_disc.step()

        print(i,
              disc_loss.cpu().detach().numpy(),
              "{0:.2f}s".format(time() - tic))
        tic = time()

        if (i % 100 == 0):
            gen.eval()
            
            save(mapping, 'saved_models/{}/mapping/{}.pt'.format(git_hash, i))
            save(gen, 'saved_models/{}/gen/{}.pt'.format(git_hash, i))            
            a, b, c = gen(z_test_c, mapping)
            a = a[0].cpu().detach().numpy().transpose(1, 0).reshape(
                -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
            b = b[0].cpu().detach().numpy().transpose(1, 0).reshape(
                -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
            c = c[0].cpu().detach().numpy().transpose(1, 0).reshape(
                -1, 3, 2).transpose(0, 2, 1).astype(np.float64)

            raster([[a, b, c]])
            directory = "outs/continuous/".format(i)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig("outs/continuous/test{:010d}.png".format(i))
            plt.close()
            gen.train()

        if (i % 10000 == 0) and (i > 0):
            gen.eval()
            a_, b_, c_ = gen(z_test, mapping)
            for j, (a, b, c) in enumerate(zip(a_, b_, c_)):
                a = a.cpu().detach().numpy().transpose(1, 0).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                b = b.cpu().detach().numpy().transpose(1, 0).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)
                c = c.cpu().detach().numpy().transpose(1, 0).reshape(
                    -1, 3, 2).transpose(0, 2, 1).astype(np.float64)

                raster([[a, b, c]])
                directory = "outs/{:010d}/".format(i)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig("outs/{:010d}/test{:04d}.png".format(i, j))
                plt.close()
            gen.train()

if __name__ == "__main__":
    app.run(main)

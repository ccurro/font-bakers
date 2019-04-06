#!/bin/python

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

from encoder import Encoder
from decoder import Decoder

from data import Dataset

import matplotlib 
matplotlib.use('Agg') 

import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def sample_qb(control_points, steps):
    """
    cp -> [batch, 3, 6, num_curves]
    """
    batch_size = control_points.shape[0]
    num_curves = control_points.shape[-1]
    cp = control_points.permute(0, 1, 3, 2).reshape(-1, 3, num_curves, 3, 2)
    steps = torch.linspace(0, 1, steps).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    P0 = cp[..., 0, :].unsqueeze(-1)
    P1 = cp[..., 1, :].unsqueeze(-1)
    P2 = cp[..., 2, :].unsqueeze(-1)

    samples = (
        P1 + (1 - steps)**2 * (P0 - P1) + steps**2 * (P2 - P1)).transpose(
            -1, -2)
    samples = samples.reshape(batch_size, 3, -1, 2).transpose(-1, -2)

    return samples

ds = Dataset("../pil/renders/160_samples/")
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
#samples, image = next(iter(dl))


enc = Encoder(pooling=nn.MaxPool2d(2,2))
dec = Decoder()
opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)

for j in range(300):
    for i, (samples, image) in enumerate(dl):
        samples = samples.cuda()
        image = image.cuda()
        
        opt.zero_grad()
        code = enc(image)
        bezier_hat = dec(code.unsqueeze(1))
        samples_hat = sample_qb(bezier_hat, 5)
        mse = (samples - samples_hat)**2
        mse = torch.mean(mse)
        print(j, i, mse.cpu().detach().numpy())
        mse.backward()
        opt.step()

        if i % 100 == 0:
            z = samples_hat[0,0].cpu().detach().numpy()
            plt.plot(z[0,:], z[1,:])
            z = samples_hat[0,1].cpu().detach().numpy()
            plt.plot(z[0,:], z[1,:])
            z = samples_hat[0,2].cpu().detach().numpy()
            plt.plot(z[0,:], z[1,:])
            plt.savefig("yo/{}_{}.png".format(j, i))
            plt.close()
        




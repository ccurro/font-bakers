#!/bin/python

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

from encoder import Encoder
from decoder import Decoder

from data import Dataset

torch.set_default_tensor_type(torch.cuda.FloatTensor)

ds = Dataset("../pil/renders/160_samples/")
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
samples, image = next(iter(dl))

image = image.cuda()
samples = samples.cuda()

enc = Encoder(pooling=nn.MaxPool2d(2,2))
dec = Decoder()
opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()))

for _ in range(100):
    opt.zero_grad()
    code = enc(image)
    samples_hat = dec(code.unsqueeze(1))
    mse = (samples - samples_hat)**2
    mse = torch.mean(mse)
    print(mse)
    mse.backward()
    opt.step()

import matplotlib.pyplot as plt

z = samples_hat[0,0].cpu().detach().numpy()
plt.plot(z[0,:], z[1,:])
plt.show()

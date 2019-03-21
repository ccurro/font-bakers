#!/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain

torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Encoder(nn.Module):
    def __init__(
            self,
    ):
        super(Encoder, self).__init__()
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(6, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=kernel_size, padding=padding)        

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class Decoder(nn.Module):
    def __init__(
            self,
    ):
        super(Decoder, self).__init__()
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(16, 6, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv1(x)

if __name__ == "__main__":
    from data import get_batch

    enc = Encoder()
    dec = Decoder()

    def pad(a,b):
        c = F.pad(a, [0,b.shape[2] - a.shape[2], 0,0,0, b.shape[0] - a.shape[0]])
        return c

    def agg(a):
        return torch.sum(a, dim=1, keepdim=True)

    optim = optim.Adam(chain(enc.parameters(), dec.parameters()), 1e-2)

    for _ in range(1000):
        first_contours, second_contours, third_contours = get_batch(32)
        optim.zero_grad()
        f_in = torch.Tensor(first_contours.transpose(0, 2, 1))
        s_in = torch.Tensor(second_contours.transpose(0, 2, 1))
        t_in = torch.Tensor(third_contours.transpose(0, 2, 1))
        
        f = enc(f_in)
        s = enc(s_in)
        t = enc(t_in)
        
        f_ = f + 0.5*pad(agg(s), f) + 0.5*pad(agg(t), f)
        s_ = s + 0.5*agg(f[s.shape[0]]) + 0.5*pad(agg(t), s)
        t_ = t + 0.5*agg(f[t.shape[0]]) + 0.5*agg(s[t.shape[0]])
        
        f_hat = dec(f_)
        s_hat = dec(s_)
        t_hat = dec(t_)
        
        loss = (0.5*(f_in-f_hat)**2).mean()
        loss += (0.5*(s_in-s_hat)**2).mean()
        loss += (0.5*(t_in-t_hat)**2).mean()
        loss.backward()
        optim.step()
        
        print(loss.cpu().detach().numpy())


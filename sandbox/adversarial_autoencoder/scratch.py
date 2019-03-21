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
        self.conv1 = nn.Conv1d(16, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=kernel_size, padding=padding)                

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class Discriminator(nn.Module):
    def __init__(
            self,
    ):
        super(Discriminator, self).__init__()
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.conv_first = nn.Conv1d(16, 16, kernel_size=kernel_size, padding=padding)
        self.conv_second = nn.Conv1d(16, 16, kernel_size=kernel_size, padding=padding)
        self.conv_third = nn.Conv1d(16, 16, kernel_size=kernel_size, padding=padding)

        self.fc = nn.Linear(16, 1)

    def forward(self, first, second, third):
        def pad(a,b):
            c = F.pad(a, [0,b.shape[1] - a.shape[1], 0, b.shape[0] - a.shape[0]])
            return c

        
        h_first = self.conv_first(first)
        h_second = self.conv_second(second)
        h_third = self.conv_third(third)

        h = torch.mean(h_first, dim=2) + pad(torch.mean(h_second, dim=2), h_first) + pad(torch.mean(h_third, dim=2), h_first)

        return self.fc(h)

if __name__ == "__main__":
    from data import get_batch

    enc = Encoder()
    dec1 = Decoder()
    dec2 = Decoder()
    dec3 = Decoder()    
    disc = Discriminator()

    def pad(a,b):
        c = F.pad(a, [0,b.shape[2] - a.shape[2], 0,0,0, b.shape[0] - a.shape[0]])
        return c

    def agg(a):
        return torch.mean(a, dim=1, keepdim=True)

    optim_auto = optim.Adam(chain(enc.parameters(), dec1.parameters(), dec2.parameters(), dec3.parameters()), 2e-3)
    optim_disc = optim.Adam(disc.parameters(), 1e-3)
    optim_gen = optim.Adam(enc.parameters(), 1e-3)    

    for _ in range(2000):
        first_contours, second_contours, third_contours = get_batch(1024)
        optim_auto.zero_grad()
        f_in = torch.Tensor(first_contours.transpose(0, 2, 1))
        s_in = torch.Tensor(second_contours.transpose(0, 2, 1))
        t_in = torch.Tensor(third_contours.transpose(0, 2, 1))
        
        f = enc(f_in)
        s = enc(s_in)
        t = enc(t_in)
        
        f_ = f + 0.5*pad(agg(s), f) + 0.5*pad(agg(t), f)
        s_ = s + 0.5*agg(f[s.shape[0]]) + 0.5*pad(agg(t), s)
        t_ = t + 0.5*agg(f[t.shape[0]]) + 0.5*agg(s[t.shape[0]])

        f_hat = dec1(f_)
        s_hat = dec2(s_)
        t_hat = dec3(t_)
        
        loss = (0.5*(f_in-f_hat)**2).mean()
        loss += (0.5*(s_in-s_hat)**2).mean()
        loss += (0.5*(t_in-t_hat)**2).mean()

        loss.backward(retain_graph=True)
        optim_auto.step()

        optim_disc.zero_grad()
        optim_gen.zero_grad()

        disc_samples = torch.sigmoid(disc(f_, s_, t_))
        disc_noise = torch.sigmoid(disc(torch.randn(f_.shape)+1, torch.randn(s_.shape)+1, torch.randn(t_.shape)+1))

        disc_loss = F.binary_cross_entropy(disc_samples, torch.zeros_like(disc_samples)).mean()
        disc_loss += F.binary_cross_entropy(disc_noise, torch.ones_like(disc_noise)).mean()

        gen_loss = F.binary_cross_entropy(disc_samples, torch.ones_like(disc_samples)).mean()

        disc_loss.backward(retain_graph=True)
        optim_disc.step()

        gen_loss.backward()
        optim_gen.step()        

        code = f_.cpu().detach().numpy()
        print(loss.cpu().detach().numpy(), code.mean(), code.var())


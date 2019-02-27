import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
'''
Shape Info
----------
Takes a sparse character vector corresponding of shape: [batch size, 70]
and a dense style vector of shape: [batch size, 100]

Outputs a vector of generated glyphs of shape: [batch size, 20, 30, 3, 2]
'''


class Rolls(nn.Module):
    def __init__(self, device, outshape=(16, 20, 30, 3, 2)):
        super(Rolls, self).__init__()
        charVecLen = 70
        styleVecLen = 100
        self.device = device
        self.fc1 = nn.Linear(charVecLen, 100)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 3600)

    def forward(self, charVec, styleVec):
        x = self.fc1(charVec)
        x = self.relu(self.bn(x))
        x = torch.cat((x, styleVec), 1)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = x.view(16, 20, 30, 3, 2)
        return x


def rolls_optimizer(net):
    '''Returns optimizer.'''
    return optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

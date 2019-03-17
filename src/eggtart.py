import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from coordconv import CoordConv
from resnet import resnet_small
from utils import rasterize


class Eggtart(nn.Module):
    def __init__(self, device, resolution=64, sigma=0.01):
        super(Eggtart, self).__init__()
        self.resnet = resnet_small(num_classes=70)
        self.resolution = resolution
        self.sigma = sigma
        self.device = device

    def forward(self, x, raster_input=False):
        if not raster_input:
            if isinstance(x, torch.Tensor):
                # x.shape = [batch_size, 20, 30, 3, 2]
                # `forward` must return shape [batch_size, 70]
                x = x.squeeze()
                batch_size = x.shape[0]
                x = x.view(batch_size, -1, 2)
                rasters = rasterize(x, device=self.device)
            elif isinstance(x, list):
                rasters = []
                # Rasterize glyph by glyph, due to variable number of curves
                for glyph in x:
                    # points.shape == [N, 2], where N = num_contours * num_curves * 3
                    points = torch.cat(
                        [curve for contour in glyph for curve in contour])
                    points = points.unsqueeze(0)
                    raster = rasterize(points, device=self.device)
                    rasters.append(raster)
                rasters = torch.cat(rasters, dim=0)
            rasters = rasters.unsqueeze(1)
        else:
            rasters = x

        clf = self.resnet(rasters)
        return clf


def eggtart_optimizer(net):
    '''Returns optimizer and number of epochs, in that order.'''
    return optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

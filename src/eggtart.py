import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from coordconv import CoordConv
from resnet import resnet_small


class Eggtart(nn.Module):
    def __init__(self, device, resolution=64, sigma=0.01):
        super(Eggtart, self).__init__()
        self.resnet = resnet_small(num_classes=70)
        self.resolution = resolution
        self.sigma = sigma
        self.device = device

        # Padding constants chosen by looking at empirical distribution of
        # coordinates of Bezier control point from real fonts
        left_pad = 0.25 * resolution
        right_pad = 1.25 * resolution
        up_pad = 0.8 * resolution
        down_pad = 0.4 * resolution
        mesh_lr = np.linspace(
            -left_pad, resolution + right_pad, num=resolution, endpoint=False)
        mesh_ud = np.linspace(
            -down_pad, resolution + up_pad, num=resolution, endpoint=False)
        XX, YY = np.meshgrid(mesh_lr, mesh_ud)
        YY = np.flip(YY)
        XX_expanded = XX[:, :, np.newaxis]
        YY_expanded = YY[:, :, np.newaxis]
        self.x_meshgrid = torch.Tensor(XX_expanded / resolution).to(device)
        self.y_meshgrid = torch.Tensor(YY_expanded / resolution).to(device)

    def rasterize(self, x):
        '''
        Simple rasterization: drop a single Gaussian at every control point.

        Parameters
        ----------
        x : [batch_size, num_control_points, 2]
            Control points of glyphs.

        Notes
        -----
        The num_contours and num_beziers dimensions have been collapsed into one
        num_control_points dimension.

        Also, we can pad with sufficiently large coordinates (e.g. 999) to
        indicate that there are no more control points: this places a Gaussian
        off-raster, which minimally affects the raster.
        '''
        batch_size = x.size()[0]
        num_samples = x.size()[1]
        x_samples = x[:, :, 0].unsqueeze(1).unsqueeze(1)
        y_samples = x[:, :, 1].unsqueeze(1).unsqueeze(1)

        x_meshgrid_expanded = self.x_meshgrid.expand(
            batch_size, self.resolution, self.resolution, num_samples)
        y_meshgrid_expanded = self.y_meshgrid.expand(
            batch_size, self.resolution, self.resolution, num_samples)

        raster = torch.exp(
            (-(x_samples - x_meshgrid_expanded)**2 -
             (y_samples - y_meshgrid_expanded)**2) / (2 * self.sigma**2))
        raster = raster.sum(dim=3)
        return raster

    def forward(self, x):
        # x.shape = [batch_size, 20, 30, 3, 2]
        # `forward` must return shape [batch_size, 70]
        x = x.squeeze()
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 2)
        x = self.rasterize(x)
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


def eggtart_optimizer(net):
    '''Returns optimizer and number of epochs, in that order.'''
    return optim.SGD(net.parameters(), lr=0.01, momentum=0.9), 3

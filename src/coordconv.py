'''
Modified from Muhammed Kocabas: https://github.com/mkocabas/CoordConv-pytorch

A PyTorch implementation of Uber CoordConv.
'''

import torch
import torch.nn as nn


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)
        ],
                        dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) +
                torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    '''
    Uber CoordConv module. May be used to subtitute conv2d modules.

    Parameters
    ----------
    in_channels, out_channels, kernel_size
        The same parameters that pytorch.nn.Conv2d takes.
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

    with_r : boolean, default False
        Whether or not to include the radial distance from the center of the
        image as a channel in the Conv2d, in addition to the xy meshgrids.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 with_r=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, *args,
                              **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

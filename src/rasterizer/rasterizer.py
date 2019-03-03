#!/bin/python3

from time import time
import numpy as np
import torch


def _sample_quadratic_bezier(control_points, steps):
    '''
    Sample a quadratic Bezier curve, given control points and steps.

    Parameters
    ----------
    control_points : [3, 2]
        The three control points of the Bezier curve.
    steps : [num_steps,]
        The steps to sample at. Usually a linspace.

    Returns
    -------
    samples : [num_samples, 2]
    '''
    P0 = control_points[0].unsqueeze(1)
    P1 = control_points[1].unsqueeze(1)
    P2 = control_points[2].unsqueeze(1)
    samples = (P1 + (1 - steps)**2 * (P0 - P1) + steps**2 * (P2 - P1))
    return torch.t(samples)


def _sample_quadratic_bezier_derivative(control_points, steps):
    '''
    Sample the derivative of a quadratic Bezier curve, given control points and steps.

    Parameters
    ----------
    control_points : [3, 2]
        The three control points of the Bezier curve.
    steps : [num_steps,]
        The steps to sample at. Usually a linspace.

    Returns
    -------
    derivative : [num_samples, 2]
    '''
    P0 = control_points[0].unsqueeze(1)
    P1 = control_points[1].unsqueeze(1)
    P2 = control_points[2].unsqueeze(1)
    derivative = 2 * steps * (P2 - P1) + 2 * (1 - steps) * (P1 - P0)
    return torch.t(derivative)


def _sample_cubic_bezier(control_points, steps):
    '''
    Sample a cubic Bezier curve, given control points and steps.

    Parameters
    ----------
    control_points : [4, 2]
        The four control points of the Bezier curve.
    steps : [num_steps,]
        The steps to sample at. Usually a linspace.

    Returns
    -------
    samples : [num_samples, 2]
    '''
    P0 = control_points[0].unsqueeze(1)
    P1 = control_points[1].unsqueeze(1)
    P2 = control_points[2].unsqueeze(1)
    P3 = control_points[3].unsqueeze(1)
    samples = ((1 - steps)**3 * P0 + 3 * steps * (1 - steps)**2 * P1 +
               3 * (1 - steps) * steps**2 * P2 + steps**3 * P2)
    return torch.t(samples)


def _sample_cubic_bezier_derivative(control_points, steps):
    '''
    Sample the derivative of a cubic Bezier curve, given control points and steps.

    Parameters
    ----------
    control_points : [4, 2]
        The four control points of the Bezier curve.
    steps : [num_steps,]
        The steps to sample at. Usually a linspace.

    Returns
    -------
    derivative : [num_samples, 2]
    '''
    P0 = control_points[0].unsqueeze(1)
    P1 = control_points[1].unsqueeze(1)
    P2 = control_points[2].unsqueeze(1)
    P3 = control_points[3].unsqueeze(1)
    derivative = (3 * (1 - steps)**2 * (P1 - P0) + 6 * steps * (1 - steps) *
                  (P2 - P1) + 3 * steps**2 * (P3 - P2))
    return torch.t(derivative)


def sample_bezier(control_points, steps):
    '''
    Sample from a Bezier curve, given its control points.

    Parameters
    ----------
    control_points : [3, 2] or [4, 2]
        The control points of the quadratic/cubic Bezier curve.
    steps : [num_steps,]
        The steps to interpolate by.

    Returns
    -------
    curve : [num_samples, 2]
    '''
    if control_points.size()[0] == 3:
        curve = _sample_quadratic_bezier(control_points, steps)
    elif control_points.size()[0] == 4:
        curve = _sample_cubic_bezier(control_points, steps)
    else:
        raise ValueError(
            'Sampling from a Bezier curve that is neither cubic nor quadratic.'
        )
    return curve


def sample_bezier_derivative(control_points, steps):
    '''
    Sample from the derivative of a Bezier curve, given its control points.

    Parameters
    ----------
    control_points : [3, 2] or [4, 2]
        The control points of the quadratic/cubic Bezier curve.
    steps : [num_steps,]
        The steps to interpolate by.

    Returns
    -------
    curve : [num_samples, 2]
    '''
    if control_points.size()[0] == 3:
        curve = _sample_quadratic_bezier_derivative(control_points, steps)
    elif control_points.size()[0] == 4:
        curve = _sample_cubic_bezier_derivative(control_points, steps)
    else:
        raise ValueError(
            'Sampling from a Bezier curve derivative that is neither cubic nor quadratic.'
        )
    return curve


class Rasterizer(torch.nn.Module):
    def __init__(self,
                 resolution=128,
                 steps=32,
                 sigma=1e-2,
                 method='base',
                 use_cuda=True):
        super(Rasterizer, self).__init__()

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif use_cuda and not torch.cuda.is_available():
            print('Rasterizer: CUDA not available. Falling back on CPU.')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.resolution = resolution
        self.sigma = sigma
        self.method = method
        self.steps = torch.linspace(0, 1, steps).to(self.device)

        # Padding constants chosen by looking at empirical distribution of
        # coordinates of Bezier control point from real fonts
        left_pad = 0.25 * self.resolution
        right_pad = 1.25 * self.resolution
        up_pad = 0.8 * self.resolution
        down_pad = 0.4 * self.resolution
        mesh_lr = np.linspace(
            -left_pad,
            self.resolution + right_pad,
            num=self.resolution,
            endpoint=False)
        mesh_ud = np.linspace(
            -down_pad,
            self.resolution + up_pad,
            num=self.resolution,
            endpoint=False)
        XX, YY = np.meshgrid(mesh_lr, mesh_ud)
        YY = np.flip(YY)
        XX_expanded = XX[:, :, np.newaxis]
        YY_expanded = YY[:, :, np.newaxis]
        self.x_meshgrid = torch.tensor(XX_expanded / self.resolution,
                requires_grad=False, dtype=torch.float).to(self.device)
        self.y_meshgrid = torch.tensor(YY_expanded / self.resolution,
                requires_grad=False, dtype=torch.float).to(self.device)

        if method == 'base':
            self.raster = self._raster_base
        if method == 'base_half':
            self.raster = self._raster_base_half

    def forward(self, control_points):
        '''
        Forward computation: rasterize a set of control points.

        Parameters
        ----------
        control_points : [batch_size, num_beziers, 3, 2] or [batch_size, num_beziers, 4, 2]
            1st dimension is the batch size.
            2nd dimension is the number of Bezier curves.
            3rd dimension is the number of control points per Bezier curve (3 for quadratic, 4 for cubic).
            4th dimension is 2 for the x and y coordinates.

        Returns
        -------
        raster : [batch_size, resolution, resolution]
            Rastered glyphs.
        '''
        samples = torch.stack([
            torch.cat(
                [sample_bezier(bezier, self.steps) for bezier in character])
            for character in control_points
        ]).to(self.device)

        derivative_samples = torch.stack([
            torch.cat([
                sample_bezier_derivative(bezier, self.steps)
                for bezier in character
            ]) for character in control_points
        ]).to(self.device)

        return self.raster(samples, derivative_samples, self.sigma)

    def _raster_base(self, samples, derivative_samples, sigma):
        '''
        Base raster method: rasterizes a curve given samples from the curve.

        Parameters
        ----------
        samples : [batch_size, num_samples, 2]
            Samples from the glyph's Bezier curves.
            Here, num_samples = num_beziers * num_steps.
        derivative_samples : [batch_size, num_samples, 2]
            Samples from the glyph's Bezier curves' derivatives, i.e. the Bezier
            speed. Here, num_samples = num_beziers * num_steps.
        sigma : float
            Standard deviation of Gaussians, normalized by the raster resolution.

        Returns
        -------
        raster : [batch_size, resolution, resolution]
            Rastered glyph.
        '''
        batch_size = samples.size()[0]
        num_samples = samples.size()[1]
        x_samples = samples[:, :, 0].unsqueeze(1).unsqueeze(1)
        y_samples = samples[:, :, 1].unsqueeze(1).unsqueeze(1)

        x_meshgrid_expanded = self.x_meshgrid.expand(
            batch_size, self.resolution, self.resolution, num_samples)
        y_meshgrid_expanded = self.y_meshgrid.expand(
            batch_size, self.resolution, self.resolution, num_samples)

        raster = torch.exp(
            (-(x_samples - x_meshgrid_expanded)**2 -
             (y_samples - y_meshgrid_expanded)**2) / (2 * sigma**2))

        speeds = torch.norm(derivative_samples, dim=2)
        raster = torch.einsum('ijkl,il->ijk', raster, speeds)
        return raster

    def _raster_base_half(self, samples, derivative_samples, sigma):
        '''
        Identical to _raster_base, but computes in half precision floating point
        numbers. Must be run on CUDA (there is poor PyTorch support for
        HalfTensors on CPU).

        Parameters
        ----------
        samples : [batch_size, num_samples, 2]
            Samples from the glyph's Bezier curves.
            Here, num_samples = num_beziers * num_steps.
        derivative_samples : [batch_size, num_samples, 2]
            Samples from the glyph's Bezier curves' derivatives, i.e. the Bezier
            speed. Here, num_samples = num_beziers * num_steps.
        sigma : float
            Standard deviation of Gaussians, normalized by the raster resolution.

        Returns
        -------
        raster : [batch_size, resolution, resolution]
            Rastered glyph.

        Notes
        -----
        https://discuss.pytorch.org/t/pytorch1-0-halftensor-support/
        '''
        batch_size = samples.size()[0]
        num_samples = samples.size()[1]
        x_samples = samples[:, :, 0].unsqueeze(1).unsqueeze(1).half()
        y_samples = samples[:, :, 1].unsqueeze(1).unsqueeze(1).half()

        x_meshgrid_expanded = self.x_meshgrid.expand(
            batch_size, self.resolution, self.resolution, num_samples).half()
        y_meshgrid_expanded = self.y_meshgrid.expand(
            batch_size, self.resolution, self.resolution, num_samples).half()

        raster = torch.exp(
            (-(x_samples - x_meshgrid_expanded)**2 -
             (y_samples - y_meshgrid_expanded)**2) / (2 * sigma**2))

        speeds = torch.norm(derivative_samples, dim=2).half()
        raster = torch.einsum('ijkl,il->ijk', raster, speeds)
        return raster.float()

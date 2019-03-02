#!/bin/python3

from absl import app, flags
from time import time
import numpy as np
import torch

FLAGS = flags.FLAGS
flags.DEFINE_boolean('disablecuda', False, 'If True, disables CUDA.')
flags.DEFINE_integer('steps', 32, 'Number of steps.')
flags.DEFINE_integer('resolution', 64, 'Resolution of raster.')
flags.DEFINE_float('sigma', 1e-2, 'Standard deviation of Gaussians.')
flags.DEFINE_enum('method', 'base', ['base', 'base_half'],
                  'Rasterization method.')
flags.DEFINE_enum('draw', 'quadratic', ['quadratic', 'cubic', 'char'],
                  'What to draw.')
flags.DEFINE_integer('passes', 1, 'Number of passes to make.')


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
                 resolution=256,
                 steps=128,
                 sigma=1e-2,
                 method='base',
                 use_cuda=True):
        super(Rasterizer, self).__init__()
        self.resolution = resolution
        self.steps = torch.linspace(0, 1, steps)
        self.sigma = sigma
        self.method = method

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif use_cuda and not torch.cuda.is_available():
            print('Rasterizer: CUDA not available. Falling back on CPU.')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        # TODO the follow line is required for the `tiled` method
        # self.gpu = torch.empty(2)
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
        self.x_meshgrid = torch.Tensor(XX_expanded / self.resolution).to(
            self.device)
        self.y_meshgrid = torch.Tensor(YY_expanded / self.resolution).to(
            self.device)

        if method == 'base':
            self.raster = self._raster_base
        if method == 'base_half':
            self.raster = self._raster_base_half
        elif method == 'shrunk':
            self.raster = self._raster_shrunk
        """
        elif method == 'bounded':
            self.raster = self._raster_bounded
        elif method == 'tiled':
            # break in to NxN tiles
            self.tiles = 4
            self.tiles_t = torch.Tensor([self.tiles]).long().to(device)
            self.chunksize = (self.resolution // self.tiles)

            # V100 has 80 sms
            self.streams = [torch.cuda.Stream() for i in range(self.tiles**2)]
            self.raster = self._raster_tiled
        """

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

    """
    def _raster_shrunk(self, samples, sigma):
        '''
        Shrunk raster method: rasterizes a curve given samples from the curve.

        Parameters
        ----------
        samples : [2, num_samples]
            Samples from the glyph's Bezier curves.
            Here, num_samples = num_beziers * num_steps.
        sigma : []
            Standard deviation of Gaussians, normalized by the raster resolution.

        Returns
        -------
        raster : [resolution, resolution]
            Rasterized glyph.
        '''
        raster = torch.zeros([self.resolution, self.resolution],
                             requires_grad=False).to(self.device)
        num_samples = samples.size()[1]

        # Block size. Next power of 2 above 2 standard deviations, in pixel space.
        block_size = 2 * int(2**np.ceil(np.log2(2 * sigma * self.resolution)))

        x_control_expanded = samples[0].to(self.device).expand(
            block_size, block_size, num_samples)
        y_control_expanded = samples[1].to(self.device).expand(
            block_size, block_size, num_samples)

        # Lower left corner of a [block_size, block_size] block centered
        # on each point of the glyph. TODO are these transposes necessary?
        blocks = torch.t(
            torch.clamp(
                (self.resolution * samples).floor().int() - block_size // 2, 0,
                self.resolution - block_size))

        x_meshgrid_ = torch.transpose(
            self.x_meshgrid.expand(num_samples, self.resolution,
                                   self.resolution), 0, 2)
        y_meshgrid_ = torch.transpose(
            self.y_meshgrid.expand(num_samples, self.resolution,
                                   self.resolution), 0, 2)

        x_meshgrid_expanded = torch.stack([
            x_meshgrid_[i:i + block_size, j:j + block_size, t]
            for t, (i, j) in enumerate(blocks)
        ],
                                          dim=2).to(self.device)
        y_meshgrid_expanded = torch.stack([
            y_meshgrid_[i:i + block_size, j:j + block_size, t]
            for t, (i, j) in enumerate(blocks)
        ],
                                          dim=2).to(self.device)

        # Compute Gaussians
        raster_ = torch.exp(
            -((x_control_expanded - x_meshgrid_expanded)**2 +
              (y_control_expanded - y_meshgrid_expanded)**2) / (2 * sigma**2))

        for t, (x, y) in enumerate(blocks):
            raster[x:x + block_size, y:y + block_size] += raster_[:, :, t]

        raster = torch.t(torch.squeeze(raster))
        return raster

    def _raster_bounded(self, glyph, sigma):
        x = glyph[0]
        y = glyph[1]

        steps = glyph.size()[1]

        xmax = torch.clamp((self.resolution * (x.max() + 3 * sigma)).ceil(), 0,
                           self.resolution).int().item()
        ymax = torch.clamp((self.resolution * (y.max() + 3 * sigma)).ceil(), 0,
                           self.resolution).int().item()
        xmin = torch.clamp((self.resolution * (x.min() - 3 * sigma)).floor(),
                           0, self.resolution).int().item()
        ymin = torch.clamp((self.resolution * (y.min() - 3 * sigma)).floor(),
                           0, self.resolution).int().item()

        w = xmax - xmin
        h = ymax - ymin

        x_ = x.to(self.device).half()
        y_ = y.to(self.device).half()

        c = torch.transpose(
            self.x_meshgrid.half().expand(steps, self.resolution,
                                          self.resolution), 0,
            2)[xmin:xmax, ymin:ymax]
        d = torch.transpose(
            self.y_meshgrid.half().expand(steps, self.resolution,
                                          self.resolution), 0,
            2)[xmin:xmax, ymin:ymax]
        raster_ = torch.exp((-(x_ - c)**2 - (y_ - d)**2) / (2 * sigma**2))
        raster_ = torch.sum(raster_, dim=2)
        raster = torch.zeros([self.resolution,
                              self.resolution]).to(self.device)
        raster[xmin:xmax, ymin:ymax] = raster_
        return torch.transpose(torch.squeeze(raster.float()), 0, 1)

    def _raster_tiled(self, glyph, sigma):
        x = glyph[0]
        y = glyph[1]

        raster = self.gpu.new_zeros(self.resolution, self.resolution)
        steps = glyph.size()[1]
        glyph.to(self.device)
        tiles = self.gpu.new_zeros(steps, steps)
        steps_ = torch.eye(steps, device=self.device)
        x_ = x.to(self.device, non_blocking=True)
        y_ = y.to(self.device, non_blocking=True)
        c = self.x_meshgrid.expand(steps, self.resolution, self.resolution)
        c = torch.transpose(c, 0, 2)
        d = torch.transpose(
            self.y_meshgrid.expand(steps, self.resolution, self.resolution), 0,
            2)

        bound = int(self.resolution * 3 * sigma)

        curve_px = (glyph * self.resolution).long().to(self.device)
        x_px, y_px = curve_px[0], curve_px[1]
        curve_tile = torch.min((curve_px / self.chunksize), self.tiles_t - 1)
        x_tile, y_tile = curve_tile[0], curve_tile[1]

        center_tiles = (self.tiles * y_tile + x_tile).long()
        right_tiles = ((x_tile < self.tiles - 1) &
                       (x_px + bound >= (x_tile + 1) * self.chunksize)).long()
        left_tiles = ((x_tile > 0) &
                      (x_px - bound < x_tile * self.chunksize)).long()
        bottom_tiles = ((y_tile < self.tiles - 1) &
                        (y_px + bound >=
                         (y_tile + 1) * self.chunksize)).long() * self.tiles
        top_tiles = (
            (y_tile > 0) &
            (y_px - bound < y_tile * self.chunksize)).long() * self.tiles

        tiles = tiles.index_add_(0, center_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + right_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles - left_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + bottom_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles - top_tiles, steps_)
        tiles = tiles.index_add_(0, center_tiles + right_tiles + bottom_tiles,
                                 steps_)
        tiles = tiles.index_add_(0, center_tiles - left_tiles + bottom_tiles,
                                 steps_)
        tiles = tiles.index_add_(0, center_tiles + right_tiles - top_tiles,
                                 steps_)
        tiles = tiles.index_add_(0, center_tiles - left_tiles - top_tiles,
                                 steps_)

        for tile, stream in enumerate(self.streams):
            with torch.cuda.stream(stream):
                steps_in_tile = tiles[tile].nonzero().reshape(-1)
                if steps_in_tile.size()[0] > 0:
                    y_tile, x_tile = divmod(tile, self.tiles)
                    x_idx = self.chunksize * x_tile
                    y_idx = self.chunksize * y_tile
                    ci = c[x_idx:x_idx + self.chunksize, y_idx:y_idx +
                           self.chunksize, steps_in_tile]
                    di = d[x_idx:x_idx + self.chunksize, y_idx:y_idx +
                           self.chunksize, steps_in_tile]
                    raster_ = torch.exp(
                        (-(x_[steps_in_tile] - ci)**2 -
                         (y_[steps_in_tile] - di)**2) / (2 * sigma**2))
                    raster_ = torch.sum(raster_, dim=2)
                    raster[x_idx:x_idx + self.chunksize, y_idx:y_idx +
                           self.chunksize] = raster_

        torch.cuda.synchronize()
        return torch.transpose(torch.squeeze(raster.float()), 0, 1)
    """


def main(argv):
    use_cuda = torch.cuda.is_available() and not FLAGS.disablecuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device "{}"'.format(device))

    # Set toy control points for testing
    if FLAGS.draw == 'quadratic':
        # Shape: [2, 1, 3, 2]
        control_points_ = np.array([[[[0.1, 0.1], [0.9, 0.9], [0.5, 0.9]]],
                                    [[[0.1, 0.1], [0.9, 0.9], [0.5, 0.9]]]])
    elif FLAGS.draw == 'cubic':
        # Shape: [1, 1, 4, 2]
        control_points_ = np.array([[[[1.0, 0.0], [0.21, 0.12], [0.72, 0.83],
                                      [0.0, 1.0]]]])
    elif FLAGS.draw == 'char':
        # Shape: [1, 3, 3, 2]
        control_points_ = np.array([[
            [[0.1, 0.1], [0.9, 0.9], [0.5, 0.9]],
            [[0.5, 0.9], [0.1, 0.9], [0.3, 0.3]],
            [[0.3, 0.3], [0.9, 0.9], [0.9, 0.1]],
        ]])
    else:
        raise ValueError(
            "`draw` flag must be one of 'quadratic', 'cubic' or 'char'.")
    control_points = torch.autograd.Variable(
        torch.Tensor(control_points_), requires_grad=True)

    rasterizer = Rasterizer(
        resolution=FLAGS.resolution,
        steps=FLAGS.steps,
        sigma=FLAGS.sigma,
        method=FLAGS.method)

    rasterizer.to(device)
    if use_cuda:
        torch.cuda.synchronize()

    elapsed_forward = 0
    elapsed_backward = 0
    memory_cached = 0
    memory_allocated = 0
    crit = torch.nn.L1Loss().cuda()

    # Time all computation
    tic_total = time()

    for i in range(FLAGS.passes):
        # Time forward pass
        tic = time()
        raster = rasterizer.forward(control_points)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_forward += time() - tic

        loss = crit(raster, raster.clone().detach())

        # Time backward pass
        tic = time()
        loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_backward += time() - tic

        # Measure allocated memory
        if use_cuda:
            memory_allocated = torch.cuda.max_memory_allocated()
            memory_cached = torch.cuda.max_memory_cached()

    elapsed = time() - tic_total

    print(
        'forwards:  {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
        .format(FLAGS.passes, elapsed_forward, FLAGS.passes / elapsed_forward,
                elapsed_forward / FLAGS.passes * 1e3))
    print(
        'backwards: {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
        .format(FLAGS.passes, elapsed_backward,
                FLAGS.passes / elapsed_backward,
                elapsed_backward / FLAGS.passes * 1e3))
    print(
        'total:     {:4d} passes in {:7.3f} seconds [{:8.3f} iter/s {:>5.1f} ms/iter].'
        .format(FLAGS.passes, elapsed, FLAGS.passes / elapsed,
                elapsed / FLAGS.passes * 1e3))
    print('memory usage:  {:5d} MB allocated, {:5d} MB cached.'.format(
        int(memory_allocated // 1e6), int(memory_cached // 1e6)))


if __name__ == '__main__':
    app.run(main)

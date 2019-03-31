#!/bin/python

import sys
import glob
import font_pb2
import random
import numpy as np

import torch
from torch.utils import data
from util import raster
from collections import deque
import matplotlib.pyplot as plt

import bezier


def pad_arrays(l):
    pad_length = max([a.shape[0] for a in l])
    out_list = []
    for a in l:
        b = np.pad(
            a, ((0, pad_length - a.shape[0]), (0, 0)),
            'constant',
            constant_values=(0.0, 0.0))
        out_list.append(b)

    return np.array(out_list)


class FontData(object):
    def __init__(self, max_num_examples, max_num_pts_per_contour):
        self.max_num_pts_per_contour = max_num_pts_per_contour
        try:
            self.protobufs = glob.glob('../data/filteredas/*')  #[:max_num_examples]
        except IndexError:
            print("Asked for too many examples in FontData Object")

    def read(self, bufs):
        glyphs = []
        for buf in bufs:
            glyph_proto = font_pb2.glyph()
            glyph_proto.ParseFromString(open(buf, 'rb').read())
            glyph = glyph_proto.glyph[0]
            bezier_points = deque(glyph.bezier_points)
            num_pts_per_contour = glyph.contour_locations

            if max(num_pts_per_contour) > self.max_num_pts_per_contour:
                #print("oof", max(num_pts_per_contour))
                continue

            contours = []

            for num_pts in num_pts_per_contour:
                pts = np.array(
                    [bezier_points.popleft() for _ in range(num_pts * 6)])
                contours.append(pts.reshape([-1, 6]))

            glyphs.append([contours])

        return glyphs

    def get_batch(self, batch_size):
        batch = self.read(
            [random.choice(self.protobufs) for _ in range(batch_size)])
        indices = [
            i[0] for i in sorted(
                enumerate(batch), key=lambda x: len(x[1]), reverse=True)
        ]

        first_contours = []
        second_contours = []
        third_contours = []

        for i in indices:
            try:
                first_contours.append(batch[i][0][0])
            except IndexError:
                continue
            try:
                second_contours.append(batch[i][0][1])
            except IndexError:
                continue
            try:
                third_contours.append(batch[i][0][2])
            except IndexError:
                continue

        return np.array(pad_arrays(first_contours)).transpose(0, 2, 1).astype(
            np.float32), np.array(pad_arrays(second_contours)).transpose(
                0, 2, 1).astype(np.float32), np.array(
                    pad_arrays(third_contours)).transpose(0, 2,
                                                          1).astype(np.float32)


class Dataset(data.Dataset):
    def __init__(self, batch_size, max_num_curves):
        self.fd = FontData(50000, max_num_curves)
        self.batch_size = batch_size

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, index):
        batch = None
        while batch is None:
            try:
                return self.fd.get_batch(self.batch_size)
            except ValueError:
                continue


def sample_qb(control_points, steps):
    steps = torch.linspace(0, 1, steps).unsqueeze(0).unsqueeze(-1)
    P0 = control_points[:, 0].unsqueeze(1)
    P1 = control_points[:, 1].unsqueeze(1)
    P2 = control_points[:, 2].unsqueeze(1)
    samples = (P1 + (1 - steps)**2 * (P0 - P1) + steps**2 * (P2 - P1))
    return samples


def _sample_quadratic_bezier_derivative(control_points, steps):
    P0 = control_points[0].unsqueeze(1)
    P1 = control_points[1].unsqueeze(1)
    P2 = control_points[2].unsqueeze(1)
    derivative = 2 * steps * (P2 - P1) + 2 * (1 - steps) * (P1 - P0)
    return torch.t(derivative)


if __name__ == "__main__":
    #print(list(map(lambda a: a.shape, get_batch(32))))

    #fd = FontData(100000, 40)
    #a, b = fd.get_batch(3)

    ds = Dataset(512, 60)
    a, b, _ = ds[0]

    raster([[
        a[0].T.reshape(-1, 3, 2).transpose(0, 2, 1),
        b[0].T.reshape(-1, 3, 2).transpose(0, 2, 1)
    ]])
    plt.figure()
    samples = sample_qb(torch.Tensor(a[0].T.reshape(-1, 3, 2)),
                        10).numpy().reshape(-1, 2)
    plt.plot(samples[:, 0], samples[:, 1], 'x')
    plt.show()
    print(a.shape)
    print(b.shape)

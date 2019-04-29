#!/bin/python

import bezier
import numpy as np
import torch
import matplotlib.pyplot as plt


def raster(glyphs):
    plots = []
    for glyph in glyphs:
        fig = plt.figure()
        ax = plt.gca()
        #plt.axis("off")
        for contour in glyph:
            for curve in contour:
                curve = bezier.Curve(curve.astype(np.float64), degree=2)
                _ = curve.plot(num_pts=256, ax=ax)

        plt.ylim([-0.5, 1.3])
        plt.xlim([-0.5, 1.3])
        fig.canvas.draw()


def lissajous_walk(num_components, max_prime=80, num_points=500, full_walk=False):
    composites = set(
        j for i in range(2, int(np.sqrt(max_prime)))
        for j in range(2 * i, 100, i))
    primes = [i for i in range(2, max_prime) if i not in composites]

    if full_walk:
        end = 2 * np.pi
    else:
        end = 0.2 * 2 * np.pi

    t = np.linspace(0, end, num=num_points)
    components = []
    for freq in primes[-num_components:]:
        sinewave = 0.5 * np.sin(freq * t) + 0.5
        components.append(sinewave)
    out = np.stack(components)
    return torch.tensor(out.T.astype(np.float32))

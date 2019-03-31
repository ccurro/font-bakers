#!/bin/python

import bezier
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

def raster(glyphs):
    plots = []
    for glyph in glyphs:
        fig  = plt.figure()
        ax = plt.gca()
        #plt.axis("off")
        for contour in glyph:
            for curve in tqdm(contour):
                curve = bezier.Curve(curve.astype(np.float64), degree=2)
                _ = curve.plot(num_pts=256, ax=ax)

        plt.ylim([-0.5, 1.3])
        plt.xlim([-0.5, 1.3])
        fig.canvas.draw()

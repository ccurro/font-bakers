#!/bin/python

import os
import glob
import numpy as np

import torch
from torch.utils import data
from imageio import imread


class Dataset(data.Dataset):
    def __init__(self, path):
        count_file = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        shards = [
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ]

        assert len(count_file) == 1

        self.path = path
        self.num_examples = int(count_file[0])
        self.num_shards = len(shards)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        image_file = os.path.join(
            self.path, "shard_{}/{}.png".format(index % self.num_shards,
                                                index))
        contour_file = os.path.join(
            self.path, "shard_{}/{}.pts.npy".format(index % self.num_shards,
                                                    index))

        return np.load(contour_file).transpose(0, 2, 1), (
            np.array(imread(image_file)).transpose(2, 0, 1) / 256.).astype(
                np.float32)


if __name__ == "__main__":
    ds = Dataset("../pil/renders/160_samples/")
    dl = data.DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for e in dl:
        print(e[0].shape)
        print(e[1].shape)
        exit()

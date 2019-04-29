#!/bin/python

import os
import glob
import numpy as np

import torch
from torch.utils import data


CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'exclam', 'numbersign', 'dollar',
    'percent', 'ampersand', 'asterisk', 'question', 'at'
]

CHARACTER_INDICES = dict(zip(CHARACTERS, list(range(len(CHARACTERS)))))


class Dataset(data.Dataset):
    def __init__(self, path, conditional=True):
        count_file = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        shards = [
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ]

        # assert len(count_file) == 1

        self.path = path
        self.conditional = conditional
        self.num_examples = 80000  # int(count_file[0])
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

        if self.conditional:
            cat_file = os.path.join(
                self.path, "shard_{}/{}.cat".format(index % self.num_shards,
                                                    index))
            with open(cat_file, 'r') as f:
                category = f.read()

            category = [CHARACTER_INDICES[c] for c in category]
        else:
            category = None

        return np.load(contour_file).transpose(0, 2, 1), category


if __name__ == "__main__":
    ds = Dataset("/zooper1/fontbakers/data/renders/160_samples_3/")
    dl = data.DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for i, (real_data, classes) in enumerate(dl):
        print(real_data.shape)
        print(len(classes))
        exit()

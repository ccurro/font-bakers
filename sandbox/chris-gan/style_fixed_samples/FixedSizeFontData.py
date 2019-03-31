#!/bin/python

import os
import glob
import numpy as np

import torch
from torch.utils import data
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, path):
        self.examples = [a for a in glob.glob(os.path.join(path, "**"), recursive=True) if os.path.isfile(a)]
        print(self.examples[0])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return np.load(self.examples[index]).transpose(0, 2, 1)


if __name__ == "__main__":
    ds = Dataset("../data/with_100_samples/")
    dl = data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for e in dl:
        print(e.shape)
        exit()

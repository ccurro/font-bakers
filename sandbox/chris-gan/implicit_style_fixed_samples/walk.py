#!/bin/python

import numpy as np

def walk(n_dims, num_samples, step=1e-1, debug=False):
    samples = np.zeros((num_samples, n_dims), dtype=np.float32)
    samples[0] = np.random.randn(n_dims)
    samples[0] /= np.sqrt((samples[0] ** 2).sum())

    for i in range(1, num_samples):
        samples[i] = samples[i-1] + np.random.randn(n_dims)*step
        samples[i] /= np.sqrt((samples[i] ** 2).sum())

        if debug:
            print(samples[i] @ samples[i-1])

    if debug:
        print(samples[0] @ samples[num_samples-1])

    return samples


if __name__ == "__main__":
    walk(10, 100, debug=True)



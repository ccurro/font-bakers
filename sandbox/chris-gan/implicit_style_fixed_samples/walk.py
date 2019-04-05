#!/bin/python

import numpy as np
from scipy.stats import multivariate_normal as mvn

# def walk(n_dims, num_samples, step=1e-1, debug=False):
#     samples = np.zeros((num_samples, n_dims), dtype=np.float32)
#     samples[0] = np.random.randn(n_dims)
#     samples[0] /= (np.sqrt((samples[0]**2).sum()) * 2)
    
#     for i in range(1, num_samples):
#         samples[i] = samples[i - 1] + np.random.randn(n_dims) * step
#         samples[i] /= (np.sqrt((samples[i]**2).sum()) * 2)

#         if debug:
#             print(samples[i])
#             print(mvn(mean=np.zeros_like(samples[i])).pdf(samples[i]))            
#             print(samples[i] @ samples[i - 1])

#     if debug:
#         print(samples[0] @ samples[num_samples - 1])

#     return samples

def  walk(num_components, num_samples=1000, max_prime=128, debug=False):
    composites = set(j for i in range(2, int(np.sqrt(max_prime))) for j in range(2*i, 200, i))
    primes = [i for i in range(2, max_prime) if i not in composites]

    t = np.linspace(0, 2*np.pi, num=num_samples)
    components = []
    for freq in primes[-num_components:]:
        sinewave = np.sin(freq * t)
        components.append(sinewave)
    out = np.stack(components)
    return out.astype(np.float32).T


if __name__ == "__main__":
    walk(64, 100, debug=True)

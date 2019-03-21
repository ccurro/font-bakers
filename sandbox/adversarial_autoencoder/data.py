#!/bin/python

import numpy as np

def get_single_example():
    num_contours = np.random.randint(1, 4)
    contours = []
    for _ in range(num_contours):
        num_curves = np.random.randint(10, 15)
        contour = np.random.uniform(size=(num_curves, 6))
        contours.append(contour)

    return contours

def pad_arrays(l):
    pad_length = max([a.shape[0] for a in l])
    out_list = []
    for a in l:
        b = np.pad(a, ((0, pad_length - a.shape[0]), (0,0)), 'constant', constant_values=(0.0, 0.0))
        out_list.append(b)

    return np.array(out_list)

def get_batch(batch_size):
    # sort batch by how many contours
    batch = [get_single_example() for _ in range(batch_size)]
    indices = [i[0] for i in sorted(enumerate(batch), key=lambda x : len(x[1]), reverse=True)]

    first_contours = []
    second_contours = []
    third_contours = []

    for i in indices:
        first_contours.append(batch[i][0])
        try:
            second_contours.append(batch[i][1])
        except IndexError:
            continue
        try:
            third_contours.append(batch[i][2])
        except IndexError:
            continue
        
    return np.array(pad_arrays(first_contours)), np.array(pad_arrays(second_contours)), np.array(pad_arrays(third_contours))


if __name__ == "__main__":
    print(list(map(lambda a: a.shape, get_batch(32))))

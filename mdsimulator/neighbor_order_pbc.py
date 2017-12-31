"""Provide the function get_neighbors.

It returns a list containing a numpy array for every cell.
The arrays themselves contain the labels of all the neighboring cells.
We account for periodic boundary conditions and avoid double counting
of interactions."""

import numpy as np
import itertools


def get_neighbors(box):
    dim = len(box.shape)       # number of dimensions
    offsets = [0, -1, 1]     # offsets, 0 first so the original entry is first
    columns = []
    # equivalent to dim nested loops over offsets
    for shift in itertools.product(offsets, repeat=dim):
        columns.append(np.roll(box, shift, np.arange(dim)).ravel())
    neighbors = np.stack(columns, axis=-1)
    # All neighbor pairs are accounted for twice, thus, we mask every
    # redudant occurence with -1
    neighbors = np.where(neighbors >= neighbors[:, [0]], neighbors, -1)
    # First column stands for the cells themselves, discard it
    # Last row stands for neighbors of the last cell, these have all been
    # accounted for already, discard it
    neighbors = neighbors[:-1, 1:]
    # In case of small dimensions (dims <= 2), we also discard double counting
    # Finally, we discard also the -1 and but all cleaned up rows in a list
    nbs = []
    for row in neighbors:
        row = np.unique(row)
        nbs.append(row[row != -1])
    return nbs

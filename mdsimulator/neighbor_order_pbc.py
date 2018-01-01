"""Provide the function get_neighbors.

It returns a list containing a numpy array for every cell.
The arrays themselves contain the labels of all the neighboring cells.
We account for periodic boundary conditions and avoid double counting
of interactions."""

import numpy as np
import itertools

def get_n_cells(box, cell_width):
    n_cells = np.floor(box / cell_width).astype(np.int)
    return n_cells

def create_cell_array(n_cells)
    total_n_cells = np.prod(n_cells)
    cell_array = np.arange(total_n_cells)
    cell_array = box.reshape(n_cells)
    return cell_array

def get_neighbors(cell_array):
    dim = len(box.shape)       # number of dimensions
    offsets = [0, -1, 1]     # offsets, 0 first so the original entry is first
    columns = []
    # equivalent to dim nested loops over offsets
    for shift in itertools.product(offsets, repeat=dim):
        columns.append(np.roll(cell_array, shift, np.arange(dim)).ravel())
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

def create_nb_order(box, cell_width):
    n_cells = get_n_cells(box, cell_width)
    cell_array = create_cell_array(n_cells)
    nbs = get_neighbors(cell_array)
    return nbs
"""Provide the function get_neighbors.

It returns a list containing a numpy array for every cell.
The arrays themselves contain the labels of all the neighboring cells.
We account for periodic boundary conditions and avoid double counting
of interactions."""

import numpy as np
import itertools

def get_n_cells(box, cell_width):
    """
    Return a numpy array with the number of cells in each dimension d.
    
    Arguments:
        box          (ndarray):      A one dimensional numpy-array with d elements  (size of preriodic box)
        cell_width   (float):        Cell width
  
    Returns:
        n_cells       (ndarray):      A one dimensional numpy-array with d elements (number of cells in each dimension d)
    """
    box = np.atleast_1d(np.asarray(box))
    n_cells = np.floor(box / cell_width).astype(np.int)
    return n_cells

def create_cell_array(n_cells):
    """
    Return a d dimensional numpy array, where each element represents one cell.
    Arguments:
        n_cells       (ndarray):      A one dimensional numpy-array with d elements (number of cells in each dimension d)
  
    Returns:
        cell_array    (ndarray):      A d dimensional array (representing all cells)
    """
    total_n_cells = np.prod(n_cells)
    cell_array = np.arange(total_n_cells)
    cell_array = cell_array.reshape(n_cells)
    return cell_array

def get_neighbors(cell_array):
    """
    Compute the neighbor cell of each cell with periodic boundary conditions.
    There is no double counting of pairs.
    
    Arguments:
        cell_array    (ndarray):      A d dimensional array (representing all cells)
  
    Returns:
        nbs    (list):      A list of n numpy arrays of different size (which contain the neighbor cells of each cell)
    """
    dim = len(cell_array.shape)
    offsets = [0, -1, 1]
    columns = []

    for shift in itertools.product(offsets, repeat=dim):
        columns.append(np.roll(cell_array, shift, np.arange(dim)).ravel())
    neighbors = np.stack(columns, axis=-1)

    neighbors = np.where(neighbors > neighbors[:, [0]], neighbors, -1)
 
    neighbors = neighbors[:, 1:]

    nbs = []
    for row in neighbors:
        row = np.unique(row)
        nbs.append(row[row != -1])
    return nbs

def create_nb_order(box, cell_width):
    """Combines the above functions get_n_cells, create_cell_array and
    get_neighbors to one."""
    n_cells = get_n_cells(box, cell_width)
    cell_array = create_cell_array(n_cells)
    nbs = get_neighbors(cell_array)
    return nbs
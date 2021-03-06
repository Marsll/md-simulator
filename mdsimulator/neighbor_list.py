import numpy as np
from .neighbor_order_pbc import get_n_cells


class NeighborList(object):
    """ Cell linked list implementation.

    The simulation box is partioned in cells of equal size.
    self.head contains a "starting" pointer to a particle in self.list.
    Every element in self.head stands for one particular cell
    In self.list the particles that correspond to the cell point to
    each other - they are linked - until there are no more.
    Then the last pointer is simply -1.
    """

    def __init__(self, box, ppos, cell_width):
        """
        Arguments:
            ppos        (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles)   
            box         (ndarray):      A one dimensional numpy-array with d elements  (size of preriodic box)
            cell_width  (float):        Cell width
        """
        self.box = np.atleast_1d(np.asarray(box))

        self.ppos = ppos
        self.n_particles = len(ppos)

        self.n_cells = get_n_cells(box, cell_width)
        self.cell_width = self.box / self.n_cells

        self.total_n_cells = np.prod(self.n_cells)

        self.head = np.zeros(self.total_n_cells, dtype=int) - 1
        self.list = np.zeros(self.n_particles, dtype=int) - 1

        self.head_old = None
        self.list_old = None

        self.construct_neighbor_list()

    def update(self, ppos, keep_old=True):
        self.ppos = ppos
        if keep_old:
            self.head_old = np.copy(self.head)
            self.list_old = np.copy(self.list)
        self.head[:] = - 1
        self.list[:] = - 1

        self.construct_neighbor_list()

    def discard_update(self):
        self.head = self.head_old
        self.list = self.list_old

    def construct_neighbor_list(self):
        for i in range(0, self.n_particles):
            # calculate the cell's index
            # by the particle's pposition
            cell_index = self.get_cell_index(i)
            # the current particle points
            # to the old head of the cell
            self.list[i] = self.head[cell_index]
            # the new head of the cell
            # is the current particle
            self.head[cell_index] = i

    def get_cell_index(self, num):
        cell_index = 0
        dims = len(self.ppos[num])
        indexes = np.floor(self.ppos[num] / self.cell_width)
        for i in range(0, dims):
            cell_index += indexes[i] * np.prod(self.n_cells[i + 1:])

        return cell_index.astype(np.int)


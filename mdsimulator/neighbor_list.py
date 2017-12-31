import numpy as np
import numpy.testing as npt


class NeighborList(object):
    """ToDo: 
    - add documentation
    - Edge cases: particles cannot be on the edges of the box
        in particular the right and upper edge
    - Questions: If the dimension of the box is not a multiple of the cell
        width, then what? Effects?
    - Do I want to pass the whole object around (or olny reference in python?)
        with all its attributes
        or rather only head and list? If yes, how to separate?
         * Consider that when updating the neighbor list, we need a lot of
         the attributes again.
    - Should we care about making anything private?
    - Working directory issues when testing
    """

    def __init__(self, dim_box, ppos, cell_width):
        self.dim_box = np.atleast_1d(np.asarray(dim_box))

        self.ppos = ppos
        self.n_particles = len(ppos)

        self.cell_width = cell_width
        self.n_cells = np.floor(self.dim_box / cell_width).astype(np.int)
        self.cell_width = self.dim_box / self.n_cells

        self.total_n_cells = np.prod(self.n_cells)

        self.head = np.zeros(self.total_n_cells, dtype=int) - 1
        self.list = np.zeros(self.n_particles, dtype=int) - 1

        self.construct_neighbor_list()

    def update(self, pppos):
        self.ppos = pppos

        self.head[:] = - 1
        self.list[:] = - 1

        self.construct_neighbor_list()

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
        #print(self.n_cells)
        # Handle case when particle is in one of the bigger cells, due to floor
        for i in range(len(indexes)):
            if indexes[i] > self.n_cells[i] - 1:
                indexes[i] = self.n_cells[i] - 1
        for i in range(0, dims):
            #print(cell_index)
            cell_index += indexes[i] * np.prod(self.n_cells[i+1:])

        return cell_index.astype(np.int)

'''
        for i in range(dims - 1, -1, -1):
            cell_index += indexes[i] * np.prod(self.n_cells[:i])
'''
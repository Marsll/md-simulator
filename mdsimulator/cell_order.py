import numpy as np
import numpy.testing as npt


def create_cell_order_2d(r_cut, dims):
    """Returns array with all neighbors of each cell"""
    size = np.empty(len(dims))
    for i, dim in enumerate(dims):
        size[i] = np.int(dim / r_cut)
    order = []
    for j in np.arange(0, size[1]):
        for i in np.arange(0, size[0]):
            index = i + j * size[0]
            nb = []
            if i + 1 < size[0]:
                nb += [index + 1]
            if j + 1 < size[1]:
                if i != 0:
                    nb += [index + size[0] - 1]
                nb += [index + size[0]]
                if i + 1 < size[0]:
                    nb += [index + size[0] + 1]
            order += [nb]
    return order


def create_cell_order_3d(r_cut, dims):
    """Returns array with all neighbors of each cell"""
    size = np.empty(len(dims))
    for i, dim in enumerate(dims):
        size[i] = np.int(dim / r_cut)
    order = []
    for k in np.arange(0, size[2]):
        for j in np.arange(0, size[1]):
            for i in np.arange(0, size[0]):
                index = i + j * size[0] + k * size[0] * size[1]
                nb = []

                if i + 1 < size[0]:
                    nb += [index + 1]

                if j + 1 < size[1]:
                    if i != 0:
                        nb += [index + size[0] - 1]
                    nb += [index + size[0]]
                    if i + 1 < size[0]:
                        nb += [index + size[0] + 1]

                if k + 1 < size[2]:
                    if j != 0:
                        if i != 0:
                            nb += [index - size[0] - 1 + size[0] * size[1]]
                        nb += [index - size[0] + size[0] * size[1]]
                        if i + 1 < size[0]:
                            nb += [index - size[0] + 1 + size[0] * size[1]]
                    if i != 0:
                        nb += [index - 1 + size[0] * size[1]]
                    nb += [index + size[0] * size[1]]
                    if i + 1 < size[0]:
                        nb += [index + 1 + size[0] * size[1]]

                    if j + 1 < size[1]:
                        if i != 0:
                            nb += [index + size[0] - 1 + size[0] * size[1]]
                        nb += [index + size[0] + size[0] * size[1]]
                        if i + 1 < size[0]:
                            nb += [index + size[0] + 1 + size[0] * size[1]]

                order += [nb]
    return order

def create_cell_order(r_cut, dims):
    if len(dims) == 2:
        return create_cell_order_2d(r_cut, dims)
    if len(dims) == 3:
        return create_cell_order_3d(r_cut, dims)

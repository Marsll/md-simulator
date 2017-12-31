import numpy as np

def create_cell_order_2d(r_cut, dims):
    """Return array with all neighbors of each cell"""
    size = np.floor(np.array(dims) / r_cut).astype(np.int)

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
    """Return array with all neighbors of each cell"""

    size = np.floor(np.array(dims) / r_cut).astype(np.int)

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

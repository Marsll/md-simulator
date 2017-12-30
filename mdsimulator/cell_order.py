import numpy as np
#import numpy.testing as npt


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


def create_cell_order_periodic(r_cut, dims):
    """Return a list with the neighbours of each cell, 
    where neighbours are not counted twice"""
    size = np.floor(np.array(dims) / r_cut).astype(np.int)
    order = []

    for k in np.arange(0, size[2]):
        for j in np.arange(0, size[1]):
            for i in np.arange(0, size[0]):
                index = i + j * size[0] + k * size[0] * size[1]
                nb = []


def create_grid(n_cells):
    grid = np.arange(np.prod(n_cells))
    grid = grid.reshape(n_cells)
    return grid


def create_periodic_cell_order(n_cells):
    grid = create_grid(n_cells)
    dimensions = len(n_cells)
    # 2D
    if dimensions == 2:
        nb_vectors = [np.array([0, 1]),
                        np.array([1, 0]),
                        np.array([1, 1]),
                        np.array([-1, 1])]

    elif dimensions == 3:
        nb_vectors_1 = [np.array([0, 1, 0]),
                        np.array([1, 0, 0]),
                        np.array([1, 1, 0]),
                        np.array([-1, 1, 0])]
        nb_vectors_2 = [np.array([i, j, 1])
                        for i in range(-1, 2, 1) for j in range(-1, 2, 1)]
        nb_vectors = nb_vectors_1 + nb_vectors_2

    n_neighbors = len(nb_vectors)
    total_n_cells = np.prod(n_cells)
    neighbors = np.zeros((total_n_cells, n_neighbors))

    for i, cell in enumerate(np.nditer(grid)):
        for j, vec in enumerate(nb_vectors):
            neighbors[i, j] = grid[i + vec]

    return neighbors

print(create_periodic_cell_order(np.array([3, 3])))



#grid(np.array( [2,4]))


# def create_periodic_cell_order(n_cells):
#     cells = np.arange(np.prod(n_cells))
#     cells_x2 = np.concatenate([cells, cells])
#     dimensions = len(n_cells)
#     # 2D
#     if dimensions == 2:
#         nb_vectors = [1,
#                       n_cells[0],
#                       n_cells[0] + 1,
#                       n_cells[0] * (n_cells[1] - 1) + 1]
#     # 3D
#     elif dimensions == 3:
#         nb_vectors = [1,  # 1
#                       n_cells[0],  # 3
#                       n_cells[0] + 1,  # 4
#                       n_cells[0] * (n_cells[1] - 1) + 1,  # 7
#                       n_cells[0] * n_cells[1],  # 9
#                       n_cells[0] * n_cells[1] + 1,  # 10
#                       n_cells[0] * (n_cells[1] + 1) - 1,  # 11
#                       n_cells[0] * (n_cells[1] + 1),  # 12
#                       n_cells[0] * (n_cells[1] + 1) + 1,  # 13
#                       n_cells[0] * (n_cells[1] + 2) - 1,  # 14
#                       2 * n_cells[0] * n_cells[1] - n_cells[0],  # 15
#                       2 * n_cells[0] * n_cells[1] - n_cells[0] + 1,  # 16
#                       2 * n_cells[0] * n_cells[1] - 1]  # 17

#     neighbours = np.zeros((len(cells), len(nb_vectors)))
#     for i, el in enumerate(cells):
#         for j, vec in enumerate(nb_vectors):
#             neighbours[i, j] = cells_x2[i + vec]
#     return neighbours

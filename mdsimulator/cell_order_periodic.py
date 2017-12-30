import numpy as np


def get_neighbours(p, exclude_p=True, shape=None):

    ndim = len(p)

    # generate an (m, ndims) array containing all combinations of 0, 1, 2
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T

    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)

    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets    # apply offsets to p

    # optional: exclude out-of-bounds indices
    # if shape is not None:
    #   valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
    #  neighbours = neighbours[valid]

    return neighbours


def fix_out_of_bounds(x, dim):
    x = x % dim
    return x

n_cells = np.array([4, 4])
n_dims = len(n_cells)
n_nbs = (3**n_dims - 1) / 2
total_n_cells = np.prod(n_cells)
box = np.arange(total_n_cells)
box = box.reshape(n_cells)

idx_list = np.zeros((total_n_cells, n_nbs))
cell_idx = np.indices(box.shape)
print(cell_idx)
for j, idx in enumerate(np.nditer(cell_idx)):
    print(idx)
    nb_idx = get_neighbours(idx, exclude_p=True, shape=box.shape)
    #nochmal angucken
    new_nb_idx = np.zeros(nb_idx.shape, dtype=np.int)
    for i, el in enumerate(nb_idx):
        new_nb_idx[i] = fix_out_of_bounds(el, box.shape)

    nbs = box[tuple(new_nb_idx.T)]
    idx_list = nbs[: len(nbs) // 2]


#print(box)
#print(len(nbs))
print(idx_list)

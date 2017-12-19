import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
import scipy.spatial.distance as dist
from lennard_jones import all_lenard_jones_forces
from cell_order import create_cell_order


def optimize(ppos, dims, r_cut, cut=.5, alpha=1, **kwargs):
    nl = NeighborList(dims, ppos, r_cut)
    nbs = create_cell_order(r_cut, dims)
    """
    forces = all_lenard_jones_forces(ppos, nl, nbs, r_cut)
    ppos_new = ppos + alpha * forces
    hard_walls(ppos_new, dims)
    alpha_vec = np.zeros(ppos.shape[1])
"""

    ppos_new = ppos
    for i in range(0, 50000):
        #ppos_old = ppos
        ppos = ppos_new
        #forces_old = forces
        forces = all_lenard_jones_forces(ppos, nl, nbs, r_cut)

        #delta_forces = forces - forces_old
        #delta_ppos = ppos - ppos_old
        """
        for i, x in enumerate(delta_ppos.T):
            numerator = np.dot(x, delta_forces.T[i])
            denominator = np.linalg.norm(delta_forces.T[i])**2
            alpha_vec[i] = numerator / denominator
                
        """

        norm = np.linalg.norm(forces, axis = 1)
        direction = forces / np.array([norm, norm]).T
        forces[norm > cut] = cut * direction[norm > cut]
        alpha_ = alpha
        ppos_new = ppos + forces * alpha_
        hard_walls(ppos_new, dims)
        nl.update(ppos)
        print(ppos)
    return ppos_new, forces

def plot_positions(ppos):
    fig = plt.figure()
    dims = ppos.shape[1]

    if dims == 3:
        ax = fig.gca(projection='3d')
        ax.scatter(*ppos.T, marker="o")
    elif dims == 2:
        plt.scatter(*ppos.T, marker="o")
    elif dims == 1:
        y = np.zeros(ppos.shape[0])
        plt.plot(*ppos.T, np.zeros_like(y), "o")


def plot_forces(ppos, forces):
    fig = plt.figure()
    dims = ppos.shape[1]

    if dims == 3:
        ax = fig.gca(projection='3d')
        ax.quiver(*ppos.T, *forces.T, length=0.1, normalize=True)
    elif dims == 2:
        plt.quiver(*ppos.T, *forces.T)
    elif dims == 1:
        plt.quiver(*ppos.T, *forces.T)


def test_plot_positions():
    ppos = np.random.randn(10, 3)
    forces = np.random.randn(10, 3)
    plot_forces(ppos, forces)
    plot_positions(ppos)
    plt.show()


def test_optimize():
    ppos = np.array([[3.1, 2.0], [7.2, 2.0]])
    plot_positions(ppos)
    dim_box = (10, 10, 10)
    #nl = NeighborList(dim_box, ppos, cell_width=1)
    finalppos, forces = optimize(ppos, dim_box, r_cut=10)

    pairwise_distances = dist.pdist(finalppos)
    ref = np.full(pairwise_distances.shape, pairwise_distances[0])

    plot_positions(finalppos)
    plot_forces(finalppos, forces)
    plt.show()
    return finalppos
    #npt.assert_allclose(pairwise_distances, ref)


def hard_walls(ppos, dims):
    ppos[ppos <= 0] = 0.1
    for i, x in enumerate(ppos.T):
        x[x > dims[i]] = dims[i]


def test_hard_walls():
    ppos = np.array([[1, 2, 3],
                     [7, -9, 10],
                     [-5, 5, 6]])
    dims = np.array([10, 2, 5])

    ref_arr = np.array([[1, 2, 3],
                        [7, 0, 5],
                        [0, 2, 5]])
    npt.assert_equal(hard_walls(ppos, dims), )


print(test_optimize())
# test_hard_walls()
# test_plot_positions()


# Avoid this!!! Use numpy, it is really powerful and helps avoiding
# if clauses and for-loops
# for pos in ppos:
#     if pos[0] < 0:
#         pos[0] = 0.1
#     if pos[0] > dims[0]:
#         pos[0] = dims[0]
#     if pos[1] < 0:
#         pos[1] = 0.1
#     if pos[1] > dims[1]:
#         pos[1] = dims[1]
# plot_positions(ppos)

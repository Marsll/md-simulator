import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
import scipy.spatial.distance as dist
#from lennard_jones.py import all_lenard_jones_forces


def optimize(ppos, nl, alpha=1, **kwargs):
    for i in range(0, 1000):
        forces = all_lenard_jones_forces(ppos, nl)
        ppos = ppos + alpha * forces
        nl.update()
    return ppos


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
    ppos = np.random.rand(3, 3) * 10
    dim_box = (10, 10, 10)
    nl = NeighborList(dim_box, ppos, cell_width=1)
    finalppos = optimize(ppos, nl)

    pairwise_distances = dist.pdist(finalppos)
    ref = np.full(pairwise_distances.shape, pairwise_distances[0])
    np.assert_allclose(pairwise_distances, ref)


test_plot_positions()

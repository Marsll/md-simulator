import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
import scipy.spatial.distance as dist
from lennard_jones import all_lenard_jones_forces
from cell_order import create_cell_order_3d, create_cell_order_2d



def optimize(ppos, dims, r_cut, alpha=1, **kwargs):
    nl = NeighborList(dims, ppos, r_cut)
    nbs = create_cell_order_2d(r_cut, dims)

    for i in range(0, 100):
        #print(i)
        #head = nl.head
        #liste = nl.list
        forces = all_lenard_jones_forces(ppos, nl, nbs, r_cut)
        ppos = ppos + alpha * forces
        for pos in ppos:
            if pos[0] < 0:
                pos[0] = 0.1
            if pos[0] > dims[0]:
                pos[0] = dims[0] 
            if pos[1] < 0:
                pos[1] = 0.1
            if pos[1] > dims[1]:
                pos[1] = dims[1] 
        #plot_positions(ppos)
        nl.update(ppos)
        print(ppos)
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
    ppos = np.array([[2.0, 2.0], [4.2, 2.0]])
    plot_positions(ppos)
    dim_box = (10, 10, 10)
    #nl = NeighborList(dim_box, ppos, cell_width=1)
    finalppos = optimize(ppos, dim_box, r_cut = 10)

    pairwise_distances = dist.pdist(finalppos)
    ref = np.full(pairwise_distances.shape, pairwise_distances[0])
    
    plot_positions(finalppos)
    plt.show()
    #npt.assert_allclose(pairwise_distances, ref)

test_optimize()

#test_plot_positions()

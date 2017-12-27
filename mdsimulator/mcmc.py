import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
import scipy.spatial.distance as dist
from lennard_jones import all_lenard_jones_potential
from cell_order import create_cell_order

def mcmc(ppos, dims, r_cut, alpha=0.1, beta=1000, steps=10000, **kwargs):
    """Returns the configuration with minimal energy using MC"""
    nl = NeighborList(dims, ppos, r_cut)
    nbs = create_cell_order(r_cut, dims)
    potential = all_lenard_jones_potential(ppos, nl, nbs, r_cut)
    
    for i in range (0, steps):
        potential_old = np.copy(potential)
        ppos_old = np.copy(ppos)
        ppos += alpha * (np.random.random(np.shape(ppos)) - 0.5)
        potential = all_lenard_jones_potential(ppos, nl, nbs, r_cut)
        if potential >= potential_old and np.exp(-(potential-potential_old) * beta) < np.random.rand():
            ppos = np.copy(ppos_old)
            potential = np.copy(potential_old)
        hard_walls(ppos, dims)
        nl.update(ppos)
        #print(potential,ppos)
    return ppos, potential

def hard_walls(ppos, dims):
    ppos[ppos <= 0] = 0.1
    for i, x in enumerate(ppos.T):
        x[x > dims[i]] = dims[i]
        
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
        



def test_optimize():
    ppos = np.array([[3.1, 2.0, 1.], [3.2, 2.0, 6], [3, 4., 2], [5.1, 5.1, 5.1]])
    plot_positions(ppos)
    dim_box = (10, 10, 10)

    finalppos, potential = mcmc(ppos, dim_box, r_cut=5)

    pairwise_distances = dist.pdist(finalppos)
    ref = np.full(pairwise_distances.shape, pairwise_distances[0])

    plot_positions(finalppos)

    plt.show()
    return finalppos, potential, pairwise_distances

def test_mcmc():
    """Three particles in a hard box."""
    ppos = np.random.random([3,3]) * 5
    plot_positions(ppos)
    dim_box = (10, 10, 10)
    potential_ref = -3.
    finalppos, potential = mcmc(ppos, dim_box, r_cut=5, steps=10000)
    np.testing.assert_almost_equal(potential, potential_ref)

test_mcmc()
    
#print(test_optimize())



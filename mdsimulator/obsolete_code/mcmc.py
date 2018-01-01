import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
import scipy.spatial.distance as dist
from lennard_jones import all_lennard_jones_potential
from cell_order import create_cell_order


def mcmc(ppos, dims, r_cut, alpha=0.1, beta=1000, steps=10000, **kwargs):
    """Return the configuration with minimal energy using MC"""
    nl = NeighborList(dims, ppos, r_cut)
    nbs = create_cell_order(r_cut, dims)
    potential = all_lennard_jones_potential(ppos, nl, nbs, r_cut)

    for i in range(0, steps):
        potential_old = np.copy(potential)
        ppos_old = np.copy(ppos)
        ppos += alpha * (np.random.random(np.shape(ppos)) - 0.5)
        #deutlich logischer hier zu machen, sonst kömnen Teilchen zeitweise außerhalb der box sein
        hard_walls(ppos, dims)
        potential = all_lennard_jones_potential(ppos, nl, nbs, r_cut)
        if potential >= potential_old and np.exp(-(potential - potential_old) * beta) < np.random.rand():
            ppos = np.copy(ppos_old)
            potential = np.copy(potential_old)
     
        nl.update(ppos)
        # print(potential,ppos)
    return ppos, potential

def test_mcmc():
    """Three particles in a hard box."""
    ppos = np.random.random([3, 3]) * 5
    plot_positions(ppos)
    dim_box = (10, 10, 10)
    potential_ref = -3.
    finalppos, potential = mcmc(ppos, dim_box, r_cut=5, steps=10000)
    np.testing.assert_almost_equal(potential, potential_ref, decimal=2)


test_mcmc()

def test_mcmc_2():
    """Four particles in a hard box. Only one cell."""
    ppos = np.random.random([4, 3]) * 5
    plot_positions(ppos)
     dim_box = (10, 10, 10)
    potential_ref = -6.
    finalppos, potential = mcmc(ppos, dim_box, r_cut=10, steps=10000)
    np.testing.assert_almost_equal(potential, potential_ref, decimal=2)
    
test_mcmc_2() 

def test_optimize():
    ppos = np.array([[3.1, 2.0, 1.], [3.2, 2.0, 6],
                     [3, 4., 2], [5.1, 5.1, 5.1]])
    plot_positions(ppos)
    dim_box = (10, 10, 10)

    finalppos, potential = mcmc(ppos, dim_box, r_cut=5)

    pairwise_distances = dist.pdist(finalppos)
    ref = np.full(pairwise_distances.shape, pairwise_distances[0])

    plot_positions(finalppos)

    plt.show()
    return finalppos, potential, pairwise_distances



    
# print(test_optimize())

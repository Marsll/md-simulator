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


def mcmc_step(ppos, dims, r_cut, nbs=None, nl=None, alpha=0.1, beta=1000,
              epot=None, **kwargs):
    if nl is None:
        nl = NeighborList(dims, ppos, r_cut)
    if nbs is None:
        nbs = create_cell_order(r_cut, dims)
    if epot is None:
        epot = all_lennard_jones_potential(ppos, nl, nbs, r_cut)

    ppos_trial = ppos + alpha * (np.random.rand(*ppos.shape) - 0.5)
    e_trial = all_lennard_jones_potential(ppos_trial, nl, nbs, r_cut)

    diff = 1000

    if e_trial < epot or np.random.rand() < np.exp(beta * (epot - e_trial)):
        hard_walls(ppos_trial, dims)
        diff = np.absolute(epot - e_trial)
        nl.update(ppos_trial)
        return ppos_trial, e_trial, diff, nbs, nl
    return ppos, epot, diff, nbs, nl


def mcmc_alternativ(ppos, dims, r_cut, alpha=0.1, beta=1000, tol=1E-8, max_steps=1000, **kwargs):
    nl = NeighborList(dims, ppos, r_cut)
    nbs = create_cell_order(r_cut, dims)
    epot = all_lennard_jones_potential(ppos, nl, nbs, r_cut)
    diff = 1000
    count = 0
    while count < max_steps and diff >= tol:
        count += 1
        ppos, epot, diff, nbs, nl = mcmc_step(
            ppos, dims, r_cut, nbs, nl, alpha, beta, epot)
        # print(potential,ppos)
    print(count)
    return ppos, epot


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
    
#test_mcmc_2() 
    
# print(test_optimize())

def test_alternives_mcmc():
    """Three particles in a hard box."""
    ppos = np.random.random([3, 3]) * 5
    plot_positions(ppos)
    dim_box = (10, 10, 10)
    finalppos, potential = mcmc_alternativ(ppos, dim_box, r_cut=5)
    plot_positions(finalppos)
    plt.show()
    print(potential, finalppos)


test_alternives_mcmc()

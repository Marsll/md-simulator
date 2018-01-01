import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
import scipy.spatial.distance as dist
from lennard_jones import all_lennard_jones_potential
from cell_order import create_cell_order




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


def mcmc(ppos, dims, r_cut, alpha=0.1, beta=1000, tol=1E-8, max_steps=1000, **kwargs):
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
    #print(count)
    return ppos, epot


def hard_walls(ppos, dims):
    ppos[ppos <= 0] = 0.1
    for i, x in enumerate(ppos.T):
        x[x > dims[i]] = dims[i]

def back_map(ppos, box):
    for i, length in enumerate(box):
        while any(ppos[:,i] >= length):
            ppos.T[i][ppos[:,i] >= length] -= length
        while any(ppos[:,i] < 0):
            ppos.T[i][ppos[:,i] < 0] += length
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




def test_mcmc():
    """Three particles in a hard box."""
    ppos = np.random.random([3, 3]) * 5
    plot_positions(ppos)
    dim_box = (10, 10, 10)
    finalppos, potential = mcmc(ppos, dim_box, r_cut=5)
    plot_positions(finalppos)
    #plt.show()
    print(potential, finalppos)


test_mcmc()

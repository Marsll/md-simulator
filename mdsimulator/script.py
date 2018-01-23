"""test 16.1.18 Lenard jones
512 particle
e=sigma=1
r_cut=3
box=[20,20,20]"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
from neighbor_order_pbc import create_nb_order
from short_ranged import potentials, pbc

def mcmc_step(ppos, params, sigma_c, box, r_cut, nbs=None, nl=None, alpha=0.1, beta=1000,
              epot=None, **kwargs):
    if nl is None:
        nl = NeighborList(box, ppos, r_cut)
    if nbs is None:
        nbs = create_nb_order(box, r_cut)
    if epot is None:
        epot = potentials(ppos, params, sigma_c, nl, nbs, r_cut, coulomb=False)

    ppos_trial = ppos + alpha * (np.random.rand(*ppos.shape) - 0.5)
    ppos_trial = back_map(ppos_trial, nl.box)
    nl.update(ppos_trial, keep_old=True)
    e_trial = potentials(ppos_trial, params, sigma_c, nl, nbs, r_cut, coulomb=False)

    diff = 1000 #fix!!!!!!

    if e_trial < epot or np.random.rand() < np.exp(beta * (epot - e_trial)):
        diff = np.absolute(epot - e_trial)
        return ppos_trial, e_trial, diff, nbs, nl
    nl.discard_update()
    return ppos, epot, diff, nbs, nl


def mcmc(ppos, params, sigma_c, box, r_cut, alpha=0.1, beta=1000000000000000, tol=1E-8,
         max_steps=100, **kwargs):
    nl = NeighborList(box, ppos, r_cut)
    nbs = create_nb_order(box, r_cut)
    epots = []
    epot = potentials(ppos, params, sigma_c, nl, nbs, r_cut, coulomb=False)
    epots.append(epot)
    diff = 1000
    count = 0
    while count < max_steps:# and diff >= tol:
        count += 1
        ppos, epot, diff, nbs, nl = mcmc_step(
            ppos, params, sigma_c, box, r_cut, nbs, nl, alpha, beta, epot)
        epots.append(epot)
    return ppos, epot, np.asarray(epots)


def back_map(ppos, box):
    for i, length in enumerate(box):
        while any(ppos[:, i] >= length):
            ppos.T[i][ppos[:, i] >= length] -= length
        while any(ppos[:, i] < 0):
            ppos.T[i][ppos[:, i] < 0] += length
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


def mcmc_sampling():
    """Particles in a periodic box."""
    dim_box = [20, 20, 20]
    N = 64
    ppos = create_uniform_dist(dim_box, N)
    plot_positions(ppos)
    plt.show()
    r_cut = 3
    #ppos = np.random.random([N, 3]) * 20
    params = np.ones(ppos.shape)
    beta = 1
    sigma_c = 1
    finalppos, potential, epots = mcmc(ppos, params, sigma_c, dim_box, r_cut,
                                       alpha=.5,
        beta=beta, max_steps=20000)
    epots = epots[:]
    plt.hist(epots, bins='auto')
    #e_arr, n_arr = boltzmann_distribution(np.min(epots), np.max(epots),beta, len(epots))
    #plt.plot(e_arr, n_arr)
    #plt.figure()
    #plt.plot(epots)
    plot_positions(finalppos)
    plt.show()
    print(epots, finalppos)
    return epots

def create_uniform_dist(box, n):
    # suppose cubic box
    nx = np.ceil(n**(1/3))
    xx = np.linspace(0, box[0], np.int(nx), endpoint=False)
    grid = np.meshgrid(xx, xx, xx, indexing='xy')
    grid = np.array(grid).T
    grid = grid.reshape(n, len(box))
    return np.array(grid)

mcmc_sampling()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from neighbor_list import NeighborList
from neighbor_order_pbc import create_nb_order
from short_ranged import potentials, pbc
from rdf import rdf


def mcmc_step(ppos, params, sigma_c, box, r_cut, nbs=None, nl=None, alpha=0.1, beta=1000,
              epot=None, **kwargs):
    if nl is None:
        nl = NeighborList(box, ppos, r_cut)
    if nbs is None:
        nbs = create_nb_order(box, r_cut)
    if epot is None:
        epot = potentials(ppos, params, sigma_c, nl, nbs, r_cut)

    ppos_trial = ppos + alpha * (np.random.rand(*ppos.shape) - 0.5)
    ppos_trial = back_map(ppos_trial, nl.box)
    nl.update(ppos_trial, keep_old=True)
    e_trial = potentials(ppos_trial, params, sigma_c, nl, nbs, r_cut)

    diff = 1000  # fix!!!!!!

    if e_trial < epot or np.random.rand() < np.exp(beta * (epot - e_trial)):
        diff = np.absolute(epot - e_trial)
        return ppos_trial, e_trial, diff, nbs, nl
    nl.discard_update()
    return ppos, epot, diff, nbs, nl


def mcmc(ppos, params, sigma_c, box, r_cut, alpha=0.1, beta=1000000000000000, tol=1E-8,
         max_steps=100, **kwargs):
    ppos_array = [ppos]
    nl = NeighborList(box, ppos, r_cut)
    nbs = create_nb_order(box, r_cut)
    epots = []
    epot = potentials(ppos, params, sigma_c, nl, nbs, r_cut)
    epots.append(epot)
    diff = 1000
    count = 0
    while count < max_steps:  # and diff >= tol:
        count += 1
        ppos, epot, diff, nbs, nl = mcmc_step(
            ppos, params, sigma_c, box, r_cut, nbs, nl, alpha, beta, epot)
        epots.append(epot)
        ppos_array.append(ppos)
    return ppos, epot, np.asarray(epots), ppos_array


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
    """Particles in a periodic box."""
    #ppos = np.random.random([3, 3]) * 5
    ppos = np.array([[0, 0, 0], [2, 0, 0], [3, 2, 1]])
    params = np.ones(ppos.shape)
    params[1, 0] = 1
    sigma_c = 1
    plot_positions(ppos)
    dim_box = (5, 5, 5)
    finalppos, potential, _ = mcmc(ppos, params, sigma_c, dim_box, r_cut=5)
    plot_positions(finalppos)
    # plt.show()
    print(potential, np.linalg.norm(pbc(finalppos[0] - finalppos[1], dim_box)),
          np.linalg.norm(pbc(finalppos[2] - finalppos[1], dim_box)),
          np.linalg.norm(pbc(finalppos[0] - finalppos[2], dim_box)))


def mcmc_sampling():
    """Particles in a periodic box."""
    N = 64
    ppos = np.random.random([N, 3]) * 10
    # plot_positions(ppos)
    dim_box = (10, 10, 10)
    params = np.ones(ppos.shape)
    params[: int(N / 2), 0] = -1
    # print(params)
    beta = 100000
    sigma_c = 1
    finalppos, potential, epots, ppos_array = mcmc(ppos, params, sigma_c, dim_box, r_cut=3,
                                                   alpha=.5,
                                                   beta=beta, max_steps=100)
    #print(ppos_array[-1])
    print(np.asarray(ppos_array[1000:]).shape)
    plt.figure()
    histo_average, bins = rdf(ppos_array, dim_box)
    print(histo_average)
    plt.plot(bins, histo_average)
    #epots = epots[:100]
    #print(ppos_array)
    #plt.hist(epots, bins='auto')
    # e_arr, n_arr = boltzmann_distribution(np.min(epots), np.max(epots),
    # beta, len(epots))
    # plt.plot(e_arr, n_arr)
    #plt.figure()
    #plt.plot(epots)
    # plot_positions(finalppos)
    plt.show()
    #print(epots, finalppos)
    return epots


def boltzmann_distribution(e_min, e_max, beta, N):
    e_arr = np.linspace(e_min, e_max, 10000)
    n_arr = N * np.exp(-beta * e_arr**2)
    return e_arr, n_arr


# test_mcmc()
mcmc_sampling()

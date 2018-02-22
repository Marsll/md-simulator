import numpy as np
from .short_ranged import potentials


def mcmc_step(ppos, params, sigma_c, box, r_cut,
              nbs, nl, epot, step_width, beta):

    ppos_trial = ppos + step_width * (np.random.rand(*ppos.shape) - 0.5)
    ppos_trial = back_map(ppos_trial, box)
    nl.update(ppos_trial, keep_old=True)
    e_trial = potentials(ppos_trial, params, sigma_c, nl, nbs, r_cut)

    if e_trial < epot or np.random.rand() < np.exp(beta * (epot - e_trial)):
        return ppos_trial, e_trial
    nl.discard_update()
    return ppos, epot


def back_map(ppos, box):
    for i, length in enumerate(box):
        while any(ppos[:, i] >= length):
            ppos.T[i][ppos[:, i] >= length] -= length
        while any(ppos[:, i] < 0):
            ppos.T[i][ppos[:, i] < 0] += length
    return ppos

import numpy as np
from .short_ranged import potentials
from .ewald import longrange, self_energy


def mcmc_step(ppos, params, sigma_c, box, r_cut, alpha, k_max,
              nbs, nl, e_short, e_long, step_width, beta):
    """One Markov chain step in the Metropolis Monte-Carlo optimization. 

    Arguments:
        ppos        (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles)   
        params      (ndarray):      A two-dimensional array with shape (n,4) (charge, epsillon, sigma, mass) prameters of all paricles 
        sigma_c     (float):        A positive float (width of the gaussian distribution used to shield the particle)
        box         (ndarray):      A one dimensional numpy-array with d elements (size of preriodic box)
        r_cut       (float):        A positive float (cutoff radius in real space)
        alpha       (float):        A positive float (standard deviation for Gaussian charge distribution in Ewald summation)
        k_max       (float):        A positive float (cutoff radius in k space)
        nbs         (list):         A list of n numpy arrays (which contain the neighbor cells of each cell)
        nl          (NeighborList): A cell linked list
        e_short     (float):        A float (short ranged potential energy of the system)
        e_long      (float):        A float (ong ranged potential energy of the system)
        step_width  (float):        A positive float (scaling factor for the trial step)
        beta        (float):        A positive float (1 / (k_B T))

    Returns:
        potential   (float):        Potential energy of the whole system (for the trial step or the old one)
    """
    epot = e_short + e_long
    ppos_trial = ppos + step_width * (np.random.rand(*ppos.shape) - 0.5)
    ppos_trial = back_map(ppos_trial, box)
    nl.update(ppos_trial, keep_old=True)

    e_short_trial = potentials(ppos_trial, params, sigma_c, nl, nbs, r_cut)

    e_long_trial = longrange(ppos, params[:, 0], box, k_max, alpha,
                             potential=True, forces=False)

    e_trial = e_short_trial + e_long_trial

    if e_trial < epot or np.random.rand() < np.exp(beta * (epot - e_trial)):
        return ppos_trial, e_short_trial, e_long_trial
    nl.discard_update()
    return ppos, e_short, e_long


def back_map(ppos, box):
    """Map particles which are outside of the box back into the box,
    using periodic boundary conditions.

    Arguments:
        ppos        (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles)   
        box         (ndarray):      A one dimensional numpy-array with d elements (size of preriodic box)

    Returns:
        ppos        (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles, which are all inside the box) 
    """

    for i, length in enumerate(box):
        while any(ppos[:, i] >= length):
            ppos.T[i][ppos[:, i] >= length] -= length
        while any(ppos[:, i] < 0):
            ppos.T[i][ppos[:, i] < 0] += length
    return ppos

"""
Functions to compute the radial distribution function.
"""

import numpy as np
from .short_ranged import pbc


def compute_distances(ppos, box):
    """Compute distances of all particles under periodic boundary conditions.

    Arguments:
        ppos       (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles)   
        box        (ndarray):      A one dimensional numpy-array with d elements  (size of preriodic box)

    Returns: 
        rs    (ndarray):    A one dimensional array with (n*(n-1)/2) elements (All distances of the particles)      
    """
    n = len(ppos)
    rs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = pbc(ppos[i] - ppos[j], box)
            r = np.linalg.norm(dist)
            rs.append(r)

    return np.asarray(rs)


def compute_distances_one_particle(ppos, box):
    """Compute distances of all particles under periodic boundary conditions.

    Arguments:
        ppos       (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles)   
        box        (ndarray):      A one dimensional numpy-array with d elements  (size of preriodic box)

    Returns: 
        rs    (ndarray):    A one dimensional array with (n*(n-1)/2) elements (All distances of the particles)      
    """
    n = len(ppos)
    rs = []
    for i in range(1, n):
        dist = pbc(ppos[i] - ppos[0], box)
        r = np.linalg.norm(dist)
        rs.append(r)

    return np.asarray(rs)



def rdf(sample, box, num_bins=50):
    """Create a histogram for the mean distances of all particles 
    over all time steps t 
    weighted with the radial distribution of an ideal gas.

    Arguments:
        sample     (ndarray):      A three-dimensional array with shape (t, n, d) (Positions of all particles)   
        box        (ndarray):      A one dimensional numpy-array with d elements  (size of periodic box)
        num_bins   (int):          Number of bins of the histogram

    Returns: 
        rdf        (ndarray):      A one dimensional array with num_bins elements (histogram for the radial distribution)
        bins       (ndarray):      A one dimensional array with num_bins elements (bins of the histogram)
    """
    t = len(sample)
    box = np.asarray(box)
    sample = np.asarray(sample)
    r_range = (0, np.amax(box) * np.sqrt(3) / 2)
    hists = []
    for pos_arr in sample:
        rs = compute_distances(pos_arr, box)
        hist, bins = np.histogram(rs, num_bins, r_range)
        hists.append(hist)
    h = np.asarray(hists)
    rdf = np.sum(h, axis=0) / t
    for i in range(len(rdf)):
        bin_volume = 4 / 3 * np.pi * (bins[i + 1]**3 - bins[i]**3)
        num_dist = len(sample[0]) * (len(sample[0]) - 1) / 2
        rdf[i] = rdf[i] / bin_volume / num_dist * np.product(box)
    bins = bins[1:]
    return rdf, bins


def rdf_unique(sample, types, box, num_bins=50):
    """Create a histogram for the mean distances of all particles 
    over all time steps t 
    weighted with the radial distribution of an ideal gas.

    Arguments:
        sample     (ndarray):      A three-dimensional array with shape (t, n, d) (Positions of all particles)  
        types      (ndarray):      A one dimensional numpy-array with n elements  (types of the particles)
        box        (ndarray):      A one dimensional numpy-array with d elements  (size of periodic box)
        num_bins   (int):          Number of bins of the histogram

    Returns:
        ty1         (string):       Particle type one
        ty2         (string):       Particle type two
        rdf1        (ndarray):      A one dimensional array with num_bins elements (histogram for the radial distribution for the first element)
        rdf2        (ndarray):      A one dimensional array with num_bins elements (histogram for the radial distribution for the second element)
        bins        (ndarray):      A one dimensional array with num_bins elements (bins of the histogram)
    """
    ty = np.unique(types)
    if len(ty) > 2:
        raise Exception("Too many different particle types. Only two different allowed.")
        
    sample_1 = sample[:,types == ty[0]]
    sample_2 = sample[:,types == ty[1]]
    rdf1, bins = rdf(sample_1, box, num_bins)
    rdf2, bins = rdf(sample_2, box, num_bins)
    return ty[0], ty[1], rdf1, rdf2, bins

def rdf_fast(sample, box, num_bins=50):
    """Create a histogram for the mean distances of one particle 
    over all time steps t 
    weighted with the radial distribution of an ideal gas.

    Arguments:
        sample     (ndarray):      A three-dimensional array with shape (t, n, d) (Positions of all particles)   
        box        (ndarray):      A one dimensional numpy-array with d elements  (size of periodic box)
        num_bins   (int):          Number of bins of the histogram

    Returns: 
        rdf        (ndarray):      A one dimensional array with num_bins elements (histogram for the radial distribution of one particle)
        bins       (ndarray):      A one dimensional array with num_bins elements (bins of the histogram)
    """
    t = len(sample)
    box = np.asarray(box)
    sample = np.asarray(sample)
    r_range = (0, np.amax(box) * np.sqrt(3) / 2)
    hists = []
    for pos_arr in sample:
        rs = compute_distances_one_particle(pos_arr, box)
        hist, bins = np.histogram(rs, num_bins, r_range)
        hists.append(hist)
    h = np.asarray(hists)
    rdf = np.sum(h, axis=0) / t
    for i in range(len(rdf)):
        bin_volume = 4 / 3 * np.pi * (bins[i + 1]**3 - bins[i]**3)
        num_dist = len(sample[0])
        rdf[i] = rdf[i] / bin_volume / num_dist * np.product(box)
    bins = bins[1:]
    return rdf, bins

def rdf_fast_unique(sample, types, box, num_bins=50):
    """Create a histogram for the mean distances of one particles 
    over all time steps t 
    weighted with the radial distribution of an ideal gas.

    Arguments:
        sample     (ndarray):      A three-dimensional array with shape (t, n, d) (Positions of all particles)  
        types      (ndarray):      A one dimensional numpy-array with n elements  (types of the particles)
        box        (ndarray):      A one dimensional numpy-array with d elements  (size of periodic box)
        num_bins   (int):          Number of bins of the histogram

    Returns:
        ty1         (string):       Particle type one
        ty2         (string):       Particle type two
        rdf1        (ndarray):      A one dimensional array with num_bins elements (histogram for the radial distribution for the first element)
        rdf2        (ndarray):      A one dimensional array with num_bins elements (histogram for the radial distribution for the second element)
        bins        (ndarray):      A one dimensional array with num_bins elements (bins of the histogram)
    """
    ty = np.unique(types)
    if len(ty) > 2:
        raise Exception("Too many different particle types. Only two different allowed.")
        
    sample_1 = sample[:,types == ty[0]]
    sample_2 = sample[:,types == ty[1]]
    rdf1, bins = rdf_fast(sample_1, box, num_bins)
    rdf2, bins = rdf_fast(sample_2, box, num_bins)
    return ty[0], ty[1], rdf1, rdf2, bins
import numpy as np


def compute_distances(ppos, box):
    n = len(ppos)
    rs = []
    for i in range(n):
        for j in range(i+1, n):
            dist = pbc(ppos[i] - ppos[j], box)
            r = np.linalg.norm(dist)
            rs.append(r)

    return np.asarray(rs)


def pbc(dist, box):
    for i, length in enumerate(box):
        while dist[i] >= 0.5 * length:
            dist[i] -= length
        while dist[i] < -0.5 * length:
            dist[i] += length
    return dist


def rdf(sample, box, num_bins=50):
    n = len(sample)
    box = np.asarray(box)
    sample = np.asarray(sample)
    r_range = (0, np.amax(box))
    hists = []
    for pos_arr in sample:
        rs = compute_distances(pos_arr, box)
        hist, bins = np.histogram(rs, num_bins, r_range)
        hists.append(hist)
    bins = bins[1:]
    h = np.asarray(hists)
    h = h / bins**2
    rdf = np.sum(h, axis=0) / n
    return rdf, bins

import numpy as np

def compute_distances(ppos, box):
    n = len(ppos)
    rs = []
    for i in range(n):
        for j in range(i, n):
            dist = pbc(ppos[i] - ppos[j], box)
            r= np.linalg.norm(dist)
            rs.append()

    return np.asarray(rs)


def pbc(dist, box):
    #box (x-length, y-length, z-length) --> enumerate -> ((1,x-length),...)
    for i, length in enumerate(box):
        while dist[i] >= 0.5 * length:
            dist[i] -= length
        while dist[i] < -0.5 * length:
            dist[i] += length
    return dist

def rdf(steps, sample):
    for 
    np.histogram()

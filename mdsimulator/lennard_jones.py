import numpy as np
from cell_order import create_cell_order_3d, create_cell_order_2d


def lenard_jones_potential(r1, r2, epsillon=1, sigma=1):
    """Computes the lennard jones potential between two particles."""
    r12 = np.linalg.norm(r1 - r2)
    rs = sigma / r12
    potential = 4 * epsillon * (rs**12 - rs**6)
    return potential


def lenard_jones_forces(r1, r2, r_cut=10, epsillon=1, sigma=1):
    """Computes the lennard jones force between two particles.
    The direction of the force is relative to particle 1 (r1)."""
    r12 = np.linalg.norm(r1 - r2)
    force = 0
    #only compute force if distance is smaller than cutoff
    if r12 < r_cut:
        rs = sigma / r12
        force = 24 * epsillon / r12 * (2 * rs**12 - rs**6)
    direction = (r1 - r2) / r12
    return force * direction


def all_lenard_jones_forces(ppos, nl, nbs, r_cut=10, epsillon=1, sigma=1):
    """Computes the resulting lenard jones forces of a certain distribution ppos
    with a neighbour list nl"""
    forces = np.zeros((ppos.shape))
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                force = lenard_jones_forces(
                    ppos[cell], ppos[next_cell], r_cut, epsillon, sigma)
                forces[cell] += force
                forces[next_cell] += - force
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    force = lenard_jones_forces(
                        ppos[cell], ppos[next_nbcell], r_cut, epsillon, sigma)
                    forces[cell] += force
                    forces[next_nbcell] += - force
                    next_nbcell = nl.list[next_nbcell]
            cell = next_cell
    return forces



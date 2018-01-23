import numpy as np


def lennard_jones_potential(r1, r2, r_cut=20, epsilon=1, sigma=1):
    """Compute the lennard jones potential between two particles."""
    r12 = np.linalg.norm(r1 - r2)
    potential = 0
    if r12 <= r_cut:
        rs = sigma / r12
        potential = 4 * epsilon * (rs**12 - rs**6)
    return potential


def lennard_jones_forces(r1, r2, r_cut=10, epsilon=1, sigma=1):
    """Compute the lennard jones force between two particles.
    The direction of the force is relative to particle 1 (r1).
    The force acts on particle 1 (r1)."""
    r12 = np.linalg.norm(r1 - r2)
    force = 0
    # only compute force if distance is smaller than cutoff
    if r12 <= r_cut:
        rs = sigma / r12
        force = 24 * epsilon / r12 * (2 * rs**12 - rs**6)
    direction = (r1 - r2) / r12
    return force * direction


def all_lennard_jones_forces(ppos, nl, nbs, r_cut=10, epsilon=1, sigma=1):
    """Compute the resulting lennard jones forces of a certain distribution ppos
    with a neighbour list nl"""
    forces = np.zeros((ppos.shape))
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                force = lennard_jones_forces(
                    ppos[cell], ppos[next_cell], r_cut, epsilon, sigma)
                forces[cell] += force
                forces[next_cell] += - force
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    force = lennard_jones_forces(
                        ppos[cell], ppos[next_nbcell], r_cut, epsilon, sigma)
                    forces[cell] += force
                    forces[next_nbcell] += - force
                    next_nbcell = nl.list[next_nbcell]
            cell = nl.list[cell]
    return forces


def all_lennard_jones_potential(ppos, nl, nbs, r_cut=20, epsilon=1, sigma=1):
    """Compute the resulting lennard jones potential of a certain distribution ppos
    with a neighbour list nl"""
    potential = 0
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                potential += lennard_jones_potential(
                    ppos[cell], ppos[next_cell], r_cut, epsilon, sigma)
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    potential += lennard_jones_potential(
                        ppos[cell], ppos[next_nbcell], r_cut, epsilon, sigma)
                    next_nbcell = nl.list[next_nbcell]
            cell = nl.list[cell]
    return potential


def lennard_jones(ppos, nl, nbs, r_cut=5, epsilon=1, sigma=1):
    """Compute the resulting lennard jones potential and forces of a 
    certain distribution ppos using the corresponding neighbour list nl 
    and cell order nbs."""

    forces = np.zeros((ppos.shape))
    potential = 0
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                force = lennard_jones_forces(
                    ppos[cell], ppos[next_cell], r_cut, epsilon, sigma)
                forces[cell] += force
                forces[next_cell] += - force
                potential += lennard_jones_potential(
                    ppos[cell], ppos[next_cell], r_cut, epsilon, sigma)
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    force = lennard_jones_forces(
                        ppos[cell], ppos[next_nbcell], r_cut, epsilon, sigma)
                    forces[cell] += force
                    forces[next_nbcell] += - force
                    potential += lennard_jones_potential(
                        ppos[cell], ppos[next_nbcell], r_cut, epsilon, sigma)
                    next_nbcell = nl.list[next_nbcell]
            cell = nl.list[cell]
    return forces, potential

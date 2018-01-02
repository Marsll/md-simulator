import numpy as np
from scipy.special import erfc

def pair_force(r1, r2, par1, par2, sigma_c, box, r_cut, lj=True, coulomb=True):
    """Compute the short ranged forces between two particles."""
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    force = 0
    if r12 <= r_cut:
        if lj:
            epsilon = calc_eps(par1[1], par2[1])
            sigma_lj = calc_sig(par2[2], par2[2])
            rs = sigma_lj / r12
            force += 24 * epsilon / r12 * (2 * rs**12 - rs**6)
        if coulomb:
            q1 = par1[0]
            q2 = par2[0]
            #Gaussian units!!!!!!!!!!!!!!!!!!!!!!!
            f1 = erfc(r12 / (np.sqrt(2) * sigma_c)) / r12 
            f2 = np.sqrt(2 / np.pi) / sigma_c * np.exp(- r12**2 / (2 * sigma_c**2))
            force += q1 * q2 / r12 * (f1 + f2)
    direction = dist / r12
    return force * direction


def pair_potential(r1, r2, par1, par2, sigma_c, box, r_cut, lj=True, coulomb=True):
    """Compute the short ranged potential between two particles."""
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    potential = 0
    if r12 <= r_cut:
        if lj:
            epsilon = calc_eps(par1[1], par2[1])
            sigma_lj = calc_sig(par2[2], par2[2])
            rs = sigma_lj / r12
            potential += 4 * epsilon * (rs**12 - rs**6)
        if coulomb:
            q1 = par1[0]
            q2 = par2[0]
            potential += q1 * q2 / r12 * erfc(r12 / (np.sqrt(2) * sigma_c)) 
    return potential

def pair_interactions(r1, r2, par1, par2, sigma_c, box, r_cut, lj=True, coulomb=True):
    """Compute the short ranged potential and forces between two particles."""
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    potential = 0
    force = 0
    if r12 <= r_cut:
        if lj:
            epsilon = calc_eps(par1[1], par2[1])
            sigma_lj = calc_sig(par2[2], par2[2])
            rs = sigma_lj / r12
            potential += 4 * epsilon * (rs**12 - rs**6)
            force += 24 * epsilon / r12 * (2 * rs**12 - rs**6)
        if coulomb:
            q1 = par1[0]
            q2 = par2[0]
            potential += q1 * q2 / r12 * erfc(r12 / (np.sqrt(2) * sigma_c))
            #Gaussian units!!!!!!!!!!!!!!!!!!!!!!!
            f1 = erfc(r12 / (np.sqrt(2) * sigma_c)) / r12 
            f2 = np.sqrt(2 / np.pi) / sigma_c * np.exp(- r12**2 / (2 * sigma_c**2))
            force += q1 * q2 / r12 * (f1 + f2)
    direction = dist / r12
    return potential, force * direction
    
    
    
def forces(ppos, params, sigma_c, nl, nbs, r_cut, lj=True, coulomb=True):
    """Compute the resulting Lennard Jones and Colomb forces 
    of a certain distribution ppos using a neighbour list nl 
    and the neighbor order nbs"""
    
    box = nl.box
    forces = np.zeros((ppos.shape))
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                force = pair_force(
                    ppos[cell], ppos[next_cell],
                    params[cell], params[next_cell],
                    sigma_c, box, r_cut, lj, coulomb)
                
                forces[cell] += force
                forces[next_cell] -= force
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    force = pair_force(
                            ppos[cell], ppos[next_nbcell], 
                            params[cell], params[next_nbcell],
                            sigma_c, box, r_cut, lj, coulomb)
                    
                    forces[cell] += force
                    forces[next_nbcell] -= force
                    next_nbcell = nl.list[next_nbcell]
            cell = nl.list[cell]
    return forces


def potentials(ppos, params, sigma_c, nl, nbs, r_cut, lj=True, coulomb=True):
    """Compute the resulting Lennard Jones and Coulomb potential 
    of a certain distribution ppos using a neighbour list nl 
    and the neighbor order nbs"""
    
    potential = 0
    box = nl.box
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                potential += pair_potential(
                    ppos[cell], ppos[next_cell],
                    params[cell], params[next_cell],
                    sigma_c, box, r_cut, lj, coulomb)
                
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    potential += pair_potential(
                        ppos[cell], ppos[next_nbcell],
                        params[cell], params[next_nbcell],
                        sigma_c, box, r_cut, lj, coulomb)
                    
                    next_nbcell = nl.list[next_nbcell]
            cell = nl.list[cell]
    return potential


def interactions(ppos, params, sigma_c, nl, nbs, r_cut, lj=True, coulomb=True):
    """Compute the resulting lennard jones potential and forces of a 
    certain distribution ppos using the corresponding neighbour list nl 
    and cell order nbs."""
    box = nl.box
    forces = np.zeros((ppos.shape))
    potential = 0
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            # own cell
            while next_cell != -1:
                pot, force = pair_interactions(
                    ppos[cell], ppos[next_cell],
                    params[cell], params[next_cell],
                    sigma_c, box, r_cut, lj, coulomb)
                
                forces[cell] += force
                forces[next_cell] -= force
                next_cell = nl.list[next_cell]
                potential += pot
                next_cell = nl.list[next_cell]
            # neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    pot, force = pair_interactions(
                        ppos[cell], ppos[next_nbcell],
                        params[cell], params[next_nbcell],
                        sigma_c, box, r_cut, lj, coulomb)
                    
                    forces[cell] += force
                    forces[next_nbcell] -= force
                    potential += pot
                    next_nbcell = nl.list[next_nbcell]
            cell = nl.list[cell]
    return forces, potential


def pbc(dist, box):
    #box (x-length, y-length, z-length) --> enumerate -> ((1,x-length),...)
    for i, length in enumerate(box):
        while dist[i] >= 0.5 * length:
            dist[i] -= length
        while dist[i] < -0.5 * length:
            dist[i] += length
    return dist

def calc_eps(e1, e2):
    """Returns the Lenard Jones epsillon of two particles."""
    return np.sqrt(e1 * e2)

def calc_sig(s1, s2):
    """Returns the Lenard Jones sigma of two particles."""
    return (s1 + s2) / 2
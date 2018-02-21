import numpy as np
from scipy.special import erfc
import scipy.constants as const
from numba import jit


@jit
def pair_force(r1, r2, par1, par2, sigma_c, box, r_cut, lj=True, coulomb=True):
    """Compute the sum of the Lennard Jones force and the short ranged part 
    of the Coulomb force between two particles.
    
    
    Arguments:
        r1      (ndarray):      Position of the first particle
        r2      (ndarray):      Position of the second particle       
        par1    (ndarray):      Prameters of the first particle (charge, epsillon, sigma, mass)
        par2    (ndarray):      Prameters of the second particle (charge, epsillon, sigma, mass)  
        sigma_c (float):        Width of the gaussian distribution used to shield the particle
        box     (ndarray):      A d-dimensional numpy-array (size of preriodic box)
        r_cut   (float):        Cutoff radius
        lj      (boolean):      If True the Lannard Jones force is calculated
        coulomb (boolean):      If True the Coulomb force is calculated
     
    Returns:
        force * direction       (ndarray):      Force acting on the first particle due to the second one
        
    """
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    force = 0
    eps=const.epsilon_0*1e-10*const.e**-2*1e6*const.physical_constants['Avogadro constant'][0]**-1
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
            force += q1 * q2 / (4 * np.pi * eps * r12) * (f1 + f2)
    direction = dist / r12
    return force * direction * 1e10

@jit
def pair_potential(r1, r2, par1, par2, sigma_c, box, r_cut, lj=True, coulomb=True):
    """Compute the sum of the Lennard Jones potential and the short ranged part 
    of the Coulomb potential between two particles.
    
    
    Arguments:
        r1      (ndarray):      Position of the first particle
        r2      (ndarray):      Position of the second particle       
        par1    (ndarray):      Prameters of the first particle (charge, epsillon, sigma, mass)
        par2    (ndarray):      Prameters of the second particle (charge, epsillon, sigma, mass)  
        sigma_c (float):        Width of the gaussian distribution used to shield the particle
        box     (ndarray):      A d-dimensional numpy-array (size of preriodic box)
        r_cut   (float):        Cutoff radius
        lj      (boolean):      If True the Lannard Jones force is calculated
        coulomb (boolean):      If True the Coulomb force is calculated
     
    Returns:
        potential       (float):     Potential fo the two particles 
    """
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    potential = 0
    eps=const.epsilon_0*1e-10*const.e**-2*1e6*const.physical_constants['Avogadro constant'][0]**-1
    if r12 <= r_cut:
        if lj:
            epsilon = calc_eps(par1[1], par2[1])
            sigma_lj = calc_sig(par2[2], par2[2])
            rs = sigma_lj / r12
            potential += 4 * epsilon * (rs**12 - rs**6)
        if coulomb:
            q1 = par1[0]
            q2 = par2[0]
            potential += q1 * q2 / (4 * np.pi * eps * r12) * erfc(r12 / (np.sqrt(2) * sigma_c)) 
    return potential


@jit
def pair_interactions(r1, r2, par1, par2, sigma_c, box, r_cut, lj=True, coulomb=True):
    """Compute the sum of the Lennard Jones force and the short ranged part 
    of the Coulomb force and the sum of the Lennard Jones potential 
    and the short ranged part of the Coulomb potential between two particles.
    
    
    Arguments:
        r1      (ndarray):      Position of the first particle
        r2      (ndarray):      Position of the second particle       
        par1    (ndarray):      Prameters of the first particle (charge, epsillon, sigma, mass)
        par2    (ndarray):      Prameters of the second particle (charge, epsillon, sigma, mass)  
        sigma_c (float):        Width of the gaussian distribution used to shield the particle
        box     (ndarray):      A d-dimensional numpy-array (size of preriodic box)
        r_cut   (float):        Cutoff radius
        lj      (boolean):      If True the Lannard Jones force is calculated
        coulomb (boolean):      If True the Coulomb force is calculated
     
    Returns:
        potential               (float):        Potential fo the two particles 
        force * direction       (ndarray):      Force acting on the first particle
         
    """
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    potential = 0
    force = 0
    eps=const.epsilon_0*1e-10*const.e**-2*1e6*const.physical_constants['Avogadro constant'][0]**-1
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
            potential += q1 * q2 / (4 * np.pi * eps * r12) * erfc(r12 / (np.sqrt(2) * sigma_c))
            #Gaussian units!!!!!!!!!!!!!!!!!!!!!!!
            f1 = erfc(r12 / (np.sqrt(2) * sigma_c)) / r12 
            f2 = np.sqrt(2 / np.pi) / sigma_c * np.exp(- r12**2 / (2 * sigma_c**2))
            force += q1 * q2 / (4 * np.pi * eps * r12) * (f1 + f2)
    direction = dist / r12
    return potential, force * direction * 1e10
    
    
@jit    
def forces(ppos, params, sigma_c, nl, nbs, r_cut, lj=True, coulomb=True):
    """Compute the resulting Lennard Jones and Colomb forces 
    of a certain particle distribution using a neighbour lists.
    
    Arguments:
        ppos    (ndarray):      Positions of all particles    
        params  (ndarray):      Prameters of all paricles (charge, epsillon, sigma, mass)
        sigma_c (float):        Width of the gaussian distribution used to shield the particle
        nl      (NeighborList): A d-dimensional numpy-array (size of preriodic box)
        nbs     (list):         A d-dimensional numpy-array (size of preriodic box)
        r_cut   (float):        Cutoff radius
        lj      (boolean):      If True the Lannard Jones force is calculated
        coulomb (boolean):      If True the Coulomb force is calculated
     
    Returns:
        potential               (float):        Potential fo the two particles 
        force * direction       (ndarray):      Force acting on the first particle
        
        """
    
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


@jit
def potentials(ppos, params, sigma_c, nl, nbs, r_cut, lj=True, coulomb=True):
    """Compute the resulting Lennard Jones and Coulomb potential 
    of a certain distribution ppos using a neighbour list nl 
    and the neighbor order nbs
    params[0] = Ladungen
    parmas[1] = epsilons lj
    params[2] = sigma lj
    """
    
    
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


@jit
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


@jit
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

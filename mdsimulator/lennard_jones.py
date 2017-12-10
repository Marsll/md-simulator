# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:31:27 2017

@author: Leon
"""

import numpy as np

def lenard_jones_potential(r1, r2, epsillon = 1, sigma = 1):
    """Computes the lennard jones potential between two particles."""
    r12 = np.linalg.norm(r1 - r2)
    rs = sigma / r12
    potential = 4 * epsillon * (rs**12 - rs**6)
    return potential

def lenard_jones_forces(r1, r2, epsillon = 1, sigma = 1):
    """Computes the lennard jones force between two particles."""
    r12 = np.linalg.norm(r1 - r2)
    print(r1 - r2)
    rs = sigma / r12
    #wirklich -??
    force = - 24 * epsillon / rs * (2 * rs**12 - rs**6)
    direction = (r2 - r1) / r12
    return force, direction




def all_lenard_jones_forces(particle_list, epsillon = 1, sigma = 1):
    return True
    

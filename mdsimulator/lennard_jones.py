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
    """Computes the lennard jones force between two particles. (Direction for particle 1.)"""
    r12 = np.linalg.norm(r1 - r2)
    rs = sigma / r12
    force = 24 * epsillon / r12 * (2 * rs**12 - rs**6)
    direction = (r1 - r2) / r12
    return force * direction

#test for direction
#np.testing.assert_array_equal(lenard_jones_forces(0, 1, epsillon = 1, sigma = 1)[1], -1)
#np.testing.assert_array_equal(lenard_jones_forces(np.array([1,2,3]), np.array([2,3,4]), epsillon = 1, sigma = 1)[1], -np.array([1,1,1]) / np.sqrt(3))

#test for potential
np.testing.assert_equal(lenard_jones_potential(0, 1, epsillon = 1, sigma = 1), 0)
np.testing.assert_equal(lenard_jones_potential(np.array([1,2,3]), np.array([2,2,3]), epsillon = 1, sigma = 1), 0)
np.testing.assert_almost_equal(lenard_jones_potential(np.array([0,4,3]), np.array([0,0,0]), epsillon = 1, sigma = 1), -0.00025598361)

#test for forces


#test forces and directions
#attrctive
np.testing.assert_almost_equal(lenard_jones_forces(0, 5, epsillon = 1, sigma = 1),  -0.00030716067 * -1 )
#zero
np.testing.assert_almost_equal(lenard_jones_forces(0, 2**(1/6), epsillon = 1, sigma = 1), 0)
#repulsive
np.testing.assert_almost_equal(lenard_jones_forces(0, 1, epsillon = 1, sigma = 1),  24 * -1 )

np.testing.assert_almost_equal(lenard_jones_forces(np.array([0,4,3]), np.array([0,0,0]), epsillon = 1, sigma = 1),
                               -0.00030716067 * np.array([0,4,3]) / 5)







def all_lenard_jones_forces(ppos, nl, epsillon = 1, sigma = 1):
    """Computes the resulting lenard jones forces of a certain distribution ppos
    with a neighbour list nl"""
    forces = np.zeros((ppos.shape))
    for cell in nl.head:
        while cell != -1:
            next_cell = nl.list[cell]
            while next_cell != -1:
                force = lenard_jones_forces(ppos[cell], ppos[next_cell], epsillon = 1, sigma = 1)
                forces[cell] += force
                forces[next_cell] += - force 
                next_cell = nl.list[next_cell]
            cell = next_cell
            
    return forces

 
class TestNl(object):
    def __init__(self):
        self.head = np.array([1])
        self.list = np.array([-1,0])
    

test = TestNl()
particle_position_test = np.array([[0,3,4], [0,0,0]])


np.testing.assert_almost_equal(all_lenard_jones_forces(particle_position_test, test, epsillon = 1, sigma = 1),
                               np.array([-0.00030716067 * np.array([0,3,4]) / 5, 0.00030716067 * np.array([0,3,4]) / 5]))
 

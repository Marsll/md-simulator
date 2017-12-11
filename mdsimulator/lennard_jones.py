import numpy as np
from cell_order import create_cell_order_3d, create_cell_order_2d

def lenard_jones_potential(r1, r2, epsillon = 1, sigma = 1):
    """Computes the lennard jones potential between two particles."""
    r12 = np.linalg.norm(r1 - r2)
    rs = sigma / r12
    potential = 4 * epsillon * (rs**12 - rs**6)
    return potential

def lenard_jones_forces(r1, r2, epsillon = 1, sigma = 1):
    """Computes the lennard jones force between two particles. 
    The direction of the force is relative to particle 1 (r1)."""
    r12 = np.linalg.norm(r1 - r2)
    rs = sigma / r12
    force = 24 * epsillon / r12 * (2 * rs**12 - rs**6)
    direction = (r1 - r2) / r12
    return force * direction


def all_lenard_jones_forces(ppos, nl, nbs, epsillon = 1, sigma = 1):
    """Computes the resulting lenard jones forces of a certain distribution ppos
    with a neighbour list nl"""
    forces = np.zeros((ppos.shape))
    for i, cell in enumerate(nl.head):
        while cell != -1:
            next_cell = nl.list[cell]
            #own cell
            while next_cell != -1:
                force = lenard_jones_forces(ppos[cell], ppos[next_cell], epsillon, sigma)
                forces[cell] += force
                forces[next_cell] += - force 
                next_cell = nl.list[next_cell]
            #neighbour cells
            for nb in nbs[i]:
                next_nbcell = nl.head[nb]
                while next_nbcell != -1:
                    force = lenard_jones_forces(ppos[cell], ppos[next_nbcell], epsillon, sigma)
                    forces[cell] += force
                    forces[next_nbcell] += - force 
                    next_nbcell = nl.list[next_nbcell]
            cell = next_cell      
    return forces






"""Tests"""

def test_potential_1d_0():
    potential = lenard_jones_potential(0, 1, epsillon = 1, sigma = 1)
    potenital_ref = 0
    np.testing.assert_equal(potential, potenital_ref)    

def test_potential_3d_0():
    potential = lenard_jones_potential(np.array([1,2,3]), np.array([2,2,3]), epsillon = 1, sigma = 1)
    potenital_ref = 0
    np.testing.assert_equal(potential, potenital_ref)    

def test_potential_3d():
    potential = lenard_jones_potential(np.array([0,4,3]), np.array([0,0,0]), epsillon = 1, sigma = 1)
    potenital_ref = -0.00025598361
    np.testing.assert_equal(potential, potenital_ref)  


test_potential_1d_0()
test_potential_3d_0()
test_potential_3d


#test forces and directions
def test_force_attractive():
    force = lenard_jones_forces(0, 5, epsillon = 1, sigma = 1)
    force_ref = -0.00030716067 * -1
    np.testing.assert_almost_equal(force, force_ref) 

def test_force_zero():
    force = lenard_jones_forces(0, 2**(1/6), epsillon = 1, sigma = 1)
    force_ref = 0
    np.testing.assert_almost_equal(force, force_ref) 


def test_force_repulsive():
    force = lenard_jones_forces(0, 1, epsillon = 1, sigma = 1)
    force_ref =  24 * -1 
    np.testing.assert_almost_equal(force, force_ref) 

def test_force_3d():
    force = lenard_jones_forces(np.array([0,4,3]), np.array([0,0,0]), epsillon = 1, sigma = 1)
    force_ref =  -0.00030716067 * np.array([0,4,3]) / 5
    np.testing.assert_almost_equal(force, force_ref) 

test_force_repulsive()
test_force_zero()
test_force_attractive()
test_force_3d()

def test_forces_2d_1cell():
    class TestNl(object):
        def __init__(self):
            self.head = np.array([1])
            self.list = np.array([-1,0])
    
    cell_order = create_cell_order_2d(1, [1, 1])   
    test = TestNl()
    particle_position_test = np.array([[3,4], [0,0]])
    
    forces = all_lenard_jones_forces(particle_position_test, test, cell_order, epsillon = 1, sigma = 1)
    forces_ref = np.array([-0.00030716067 * np.array([3,4]) / 5, 0.00030716067 * np.array([3,4]) / 5])
    
    np.testing.assert_almost_equal(forces, forces_ref)
 


def test_forces_2d():
    class TestNl(object):
        """four cells", two particles"""
        def __init__(self):
            self.head = np.array([0, -1, -1, 1])
            self.list = np.array([-1,-1])
    
    cell_order = create_cell_order_2d(2, [4, 4])   
    
    test = TestNl()
    particle_position_test = np.array([[0,0], [3,4]])
    
    forces = all_lenard_jones_forces(particle_position_test, test, cell_order, epsillon = 1, sigma = 1)
    forces_ref = np.array([0.00030716067 * np.array([3,4]) / 5, - 0.00030716067 * np.array([3,4]) / 5])
    
    np.testing.assert_almost_equal(forces, forces_ref)
    

   
def test_forces_3d():
    """8 cells, two particles"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([-1, -1, -1, -1, 0, -1, 1, -1])
            self.list = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
    
    cell_order = create_cell_order_3d(2, [4, 4, 4])   
    
    test = TestNl()
    particle_position_test = np.array([[0, 0, 1.5], [3, 4, 1.5]])
    
    forces = all_lenard_jones_forces(particle_position_test, test, cell_order, epsillon = 1, sigma = 1)
    forces_ref = np.array([0.00030716067 * np.array([3, 4, 0]) / 5, - 0.00030716067 * np.array([3, 4, 0]) / 5])
    
    np.testing.assert_almost_equal(forces, forces_ref)

test_forces_2d_1cell()
test_forces_2d()
test_forces_3d()


print("tests done")
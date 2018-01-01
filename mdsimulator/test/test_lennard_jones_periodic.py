from ..lennard_jones_periodic import lennard_jones_forces, lennard_jones_potential, all_lennard_jones_forces, all_lennard_jones_potential, lennard_jones
from ..neighbor_order_pbc import create_nb_order
import numpy as np
import numpy.testing as npt

def test_potential_1d_0():
    pos1 = np.array([0])
    pos2 = np.array([4])
    box = (5,)
    potential = lennard_jones_potential(pos1, pos2, box, r_cut=5, epsilon=1, sigma=1)
    potenital_ref = 0
    npt.assert_equal(potential, potenital_ref)
    
    
def test_potential_3d():
    pos1 = np.array([6, 7, 0])
    pos2 = np.array([0, 0, 0])
    box = (10, 10, 10)
    potential = lennard_jones_potential(pos1, pos2, box, epsilon=1, sigma=1)
    potenital_ref = -0.00025598361
    npt.assert_almost_equal(potential, potenital_ref)

    
test_potential_1d_0()
test_potential_3d()

def test_force_3d():
    pos1 = np.array([0, 6, 7])
    pos2 = np.array([0, 0, 0])
    box = (10, 10, 10)
    force = lennard_jones_forces(pos1, pos2, box, epsilon=1, sigma=1)
    force_ref = 0.00030716067 * np.array([0, 4, 3]) / 5
    npt.assert_almost_equal(force, force_ref)

test_force_3d()



def test_potential_3d_three_particles():
    """8 cells, three particles, all in same octant"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([2, -1, -1, -1, -1, -1, -1, -1])
            self.list = np.array([-1, 0, 1])

    cell_order = create_nb_order([10, 10, 10], 5)

    test = TestNl()
    ppos = np.array([[0, 0, 0], [3, 4, 0], [4, 3, 0]])

    potential = all_lennard_jones_potential(
        ppos, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    potential_ref = 2 * (-0.00025598361) - 0.4375
    npt.assert_almost_equal(potential, potential_ref)



def test_potential_3d_four_particles():
    """8 cells, four particles, all in different octants"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([0, 1, 2, 3, -1, -1, -1, -1])
            self.list = np.array([-1, -1, -1, -1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array(
        [[4.5, 4.5, 4.5], [5.5, 4.5, 4.5], [4.5, 5.5, 4.5], [5.5, 5.5, 4.5]])

    potential = all_lennard_jones_potential(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    potential_ref = 2 * (- 0.4375)
    npt.assert_almost_equal(potential, potential_ref)


test_potential_3d_three_particles()
#test_potential_3d_four_particles()
from ..lennard_jones import lennard_jones_forces, lennard_jones_potential, all_lennard_jones_forces, all_lennard_jones_potential, lennard_jones
from ..cell_order import create_cell_order_3d, create_cell_order_2d, create_cell_order
import numpy as np
import numpy.testing as npt


"""Tests"""


def test_potential_1d_0():
    potential = lennard_jones_potential(0, 1, epsilon=1, sigma=1)
    potenital_ref = 0
    npt.assert_equal(potential, potenital_ref)


def test_potential_3d_0():
    potential = lennard_jones_potential(
        np.array([1, 2, 3]), np.array([2, 2, 3]), epsilon=1, sigma=1)
    potenital_ref = 0
    npt.assert_equal(potential, potenital_ref)


def test_potential_3d():
    potential = lennard_jones_potential(
        np.array([0, 4, 3]), np.array([0, 0, 0]), epsilon=1, sigma=1)
    potenital_ref = -0.00025598361
    npt.assert_almost_equal(potential, potenital_ref)


test_potential_1d_0()
test_potential_3d_0()
test_potential_3d


# test forces and directions
def test_force_attractive():
    force = lennard_jones_forces(0, 5, epsilon=1, sigma=1)
    force_ref = -0.00030716067 * -1
    npt.assert_almost_equal(force, force_ref)


def test_force_zero():
    force = lennard_jones_forces(0, 2**(1 / 6), epsilon=1, sigma=1)
    force_ref = 0
    npt.assert_almost_equal(force, force_ref)


def test_force_repulsive():
    force = lennard_jones_forces(0, 1, epsilon=1, sigma=1)
    force_ref = 24 * -1
    npt.assert_almost_equal(force, force_ref)


def test_force_3d():
    force = lennard_jones_forces(
        np.array([0, 4, 3]), np.array([0, 0, 0]), epsilon=1, sigma=1)
    force_ref = -0.00030716067 * np.array([0, 4, 3]) / 5
    npt.assert_almost_equal(force, force_ref)


test_force_repulsive()
test_force_zero()
test_force_attractive()
test_force_3d()


def test_forces_2d_1cell():
    class TestNl(object):
        def __init__(self):
            self.head = np.array([1])
            self.list = np.array([-1, 0])

    cell_order = create_cell_order_2d(1, [1, 1])
    test = TestNl()
    particle_position_test = np.array([[3, 4], [0, 0]])

    forces = all_lennard_jones_forces(
        particle_position_test, test, cell_order, epsilon=1, sigma=1)
    forces_ref = np.array(
        [-0.00030716067 * np.array([3, 4]) / 5,
         0.00030716067 * np.array([3, 4]) / 5])

    npt.assert_almost_equal(forces, forces_ref)


def test_forces_2d():
    class TestNl(object):
        """four cells", two particles"""

        def __init__(self):
            self.head = np.array([0, -1, -1, 1])
            self.list = np.array([-1, -1])

    cell_order = create_cell_order_2d(2, [4, 4])

    test = TestNl()
    particle_position_test = np.array([[0, 0], [3, 4]])

    forces = all_lennard_jones_forces(
        particle_position_test, test, cell_order, epsilon=1, sigma=1)
    forces_ref = np.array(
        [0.00030716067 * np.array([3, 4]) / 5,
         - 0.00030716067 * np.array([3, 4]) / 5])

    npt.assert_almost_equal(forces, forces_ref)


def test_forces_3d():
    """8 cells, two particles"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([-1, -1, -1, -1, 0, -1, 1, -1])
            self.list = np.array([-1, -1, -1, -1, -1, -1, -1, -1])

    cell_order = create_cell_order(2, [4, 4, 4])

    test = TestNl()
    particle_position_test = np.array([[0, 0, 1.5], [3, 4, 1.5]])

    forces = all_lennard_jones_forces(
        particle_position_test, test, cell_order, epsilon=1, sigma=1)
    forces_ref = np.array([0.00030716067 * np.array([3, 4, 0]) /
                           5, - 0.00030716067 * np.array([3, 4, 0]) / 5])

    npt.assert_almost_equal(forces, forces_ref)


def test_forces_3d_three_particles():
    """8 cells, three particles, all in same octant"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([2, -1, -1, -1, -1, -1, -1, -1])
            self.list = np.array([-1, 0, 1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[0, 0, 0], [3, 4, 0], [4, 3, 0]])

    forces = all_lennard_jones_forces(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    forces_ref = np.array([0.00030716067 * np.array([3, 4, 0]) / 5
                           + 0.00030716067 *
                           np.array([4, 3, 0]) / 5, - 0.00030716067 *
                           np.array([3, 4, 0]) / 5
                           + 1.5909902576697312 *
                           np.array([1, -1, 0]) / np.sqrt(2), -
                           0.00030716067 * np.array([4, 3, 0]) / 5
                           - 1.5909902576697312 * np.array([1, -1, 0]) / np.sqrt(2)])

    npt.assert_almost_equal(forces, forces_ref)


def test_forces_3d_cutoff():
    """8 cells, three particles, all in different octants 
    and out off cuttoff radius"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([0, -1, 1, -1, -1, -1, -1, 2])
            self.list = np.array([-1, -1, -1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[0, 0, 0], [10, 10, 10], [0, 10, 0]])

    forces = all_lennard_jones_forces(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    forces_ref = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    npt.assert_almost_equal(forces, forces_ref)


def test_forces_3d_three_particles_2():
    """8 cells, three particles, all in different octants"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([0, 2, 1, -1, -1, -1, -1, -1])
            self.list = np.array([-1, -1, -1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[2, 2, 2], [5, 6, 2], [6, 5, 2]])

    forces = all_lennard_jones_forces(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    forces_ref = np.array([0.00030716067 * np.array([3, 4, 0]) / 5
                           + 0.00030716067 *
                           np.array([4, 3, 0]) / 5, - 0.00030716067 *
                           np.array([3, 4, 0]) / 5
                           + 1.5909902576697312 *
                           np.array([1, -1, 0]) / np.sqrt(2), -
                           0.00030716067 * np.array([4, 3, 0]) / 5
                           - 1.5909902576697312 * np.array([1, -1, 0]) / np.sqrt(2)])

    npt.assert_almost_equal(forces, forces_ref)


test_forces_2d_1cell()
test_forces_2d()
test_forces_3d()
test_forces_3d_three_particles()
test_forces_3d_cutoff()
test_forces_3d_three_particles_2()


def test_potential_3d_three_particles():
    """8 cells, three particles, all in same octant"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([2, -1, -1, -1, -1, -1, -1, -1])
            self.list = np.array([-1, 0, 1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[0, 0, 0], [3, 4, 0], [4, 3, 0]])

    potential = all_lennard_jones_potential(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    potential_ref = 2 * (-0.00025598361) - 0.4375
    npt.assert_almost_equal(potential, potential_ref)


def test_potential_3d_three_particles_cutoff():
    """8 cells, three particles, all in different octants 
    and out off cuttoff radius."""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([0, 1, 2, -1, -1, -1, -1, -1])
            self.list = np.array([-1, -1, -1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]])

    potential = all_lennard_jones_potential(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    potential_ref = 0
    npt.assert_almost_equal(potential, potential_ref)


def test_potential_3d_three_particles_2():
    """8 cells, three particles, all in different octants"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([0, 2, 1, -1, -1, -1, -1, -1])
            self.list = np.array([-1, -1, -1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[2, 2, 2], [5, 6, 2], [6, 5, 2]])

    potential = all_lennard_jones_potential(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
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
test_potential_3d_three_particles_cutoff()
test_potential_3d_three_particles()
test_potential_3d_three_particles_2()
test_potential_3d_four_particles()


def test_forces_potential_3d_three_particles():
    """8 cells, three particles, all in different octants"""
    class TestNl(object):
        def __init__(self):
            self.head = np.array([0, 2, 1, -1, -1, -1, -1, -1])
            self.list = np.array([-1, -1, -1])

    cell_order = create_cell_order(5, [10, 10, 10])

    test = TestNl()
    particle_position_test = np.array([[2, 2, 2], [5, 6, 2], [6, 5, 2]])

    forces, potential = lennard_jones(
        particle_position_test, test, cell_order, r_cut=5, epsilon=1, sigma=1)
    forces_ref = np.array([0.00030716067 * np.array([3, 4, 0]) / 5
                           + 0.00030716067 *
                           np.array([4, 3, 0]) / 5, - 0.00030716067 *
                           np.array([3, 4, 0]) / 5
                           + 1.5909902576697312 *
                           np.array([1, -1, 0]) / np.sqrt(2), -
                           0.00030716067 * np.array([4, 3, 0]) / 5
                           - 1.5909902576697312 * np.array([1, -1, 0]) / np.sqrt(2)])
    potential_ref = 2 * (-0.00025598361) - 0.4375

    npt.assert_almost_equal(potential, potential_ref)
    npt.assert_almost_equal(forces, forces_ref)


test_forces_potential_3d_three_particles()


print("tests done")

"""tests for short_ranged for the lennard jones forces and potential"""

from ..neighbor_order_pbc import create_nb_order
from ..neighbor_list import NeighborList
from ..short_ranged import pair_potential, pair_force, potentials
import numpy as np
import numpy.testing as npt
import scipy.constants as const


def test_potential_1d_0():
    pos1 = np.array([0])
    pos2 = np.array([4])
    box = (5,)
    par1 = np.ones(3)
    par2 = np.ones(3)
    sigma_c = 1
    potential = pair_potential(pos1, pos2, par1 ,par2 ,sigma_c, box, r_cut=5, coulomb=False)
    potenital_ref = 0
    npt.assert_equal(potential, potenital_ref)
    
    
def test_potential_3d():
    pos1 = np.array([6, 7, 0])
    pos2 = np.array([0, 0, 0])
    box = (10, 10, 10)
    par1 = np.ones(3)
    par2 = np.ones(3)
    sigma_c = 1
    potential = pair_potential(pos1, pos2, par1 ,par2 ,sigma_c, box, r_cut=10, coulomb=False)
    potenital_ref = -0.00025598361
    npt.assert_almost_equal(potential, potenital_ref)

    

def test_force_3d():
    pos1 = np.array([0, 6, 7])
    pos2 = np.array([0, 0, 0])
    box = (10, 10, 10)
    par1 = np.ones(3)
    par2 = np.ones(3)
    sigma_c = 1
    force = pair_force(pos1, pos2, par1 ,par2 ,sigma_c, box, r_cut=10, coulomb=False) / 1e10
    force_ref = 0.00030716067 * np.array([0, 4, 3]) / 5
    npt.assert_almost_equal(force, force_ref)




def test_potential_3d_three_particles():
    """8 cells, three particles, all in same octant"""
    box = [10, 10, 10]
    r_cut = 5 
    nb_order = create_nb_order(box, r_cut)

    ppos = np.array([[0, 0, 0], [3, 4, 0], [6, 7, 0]])
    
    nl = NeighborList(box, ppos, r_cut)
    params = np.ones([3,3])
    sigma_c = 1
    potential = potentials(
        ppos, params, sigma_c, nl, nb_order, r_cut, lj=True, coulomb=False)
    potential_ref = 2 * 4 * ((1 / 5)**12 - (1 / 5)**6) + 4 * ((1 / np.sqrt(18))**12 - (1 / np.sqrt(18))**6)
    npt.assert_almost_equal(potential, potential_ref)
    
def test_potential_2d_three_particles_2():
    """12 cells, three particles"""
    box = [3, 4]
    r_cut = 1 
    nb_order = create_nb_order(box, r_cut)

    ppos = np.array([[0, 0], [1.5, 2.5], [2.5, 3.5]])
    
    nl = NeighborList(box, ppos, r_cut)

    params = np.ones([3,3])
    sigma_c = 1
    potential = potentials(
        ppos, params, sigma_c, nl, nb_order, r_cut, lj=True, coulomb=False)
    pot1 = 4 * ((1 / np.sqrt(0.5**2 + 0.5**2))**12 - (1 / np.sqrt(0.5**2 + 0.5**2))**6) 
    potential_ref = pot1
    npt.assert_almost_equal(potential, potential_ref)
    
def test_potential_2d_four_particles_3():
    """12 cells, no neighbours."""
    box = [15, 20]
    r_cut = 5 
    nb_order = create_nb_order(box, r_cut)

    ppos = np.array([[0, 0], [0, 11], [9, 19], [9, 6]])
    
    nl = NeighborList(box, ppos, r_cut)

    params = np.ones([4,3])
    sigma_c = 1
    potential = potentials(
        ppos, params, sigma_c, nl, nb_order, r_cut, lj=True, coulomb=False)
    potential_ref = 0
    npt.assert_almost_equal(potential, potential_ref)
    
def test_potential_3d_four_particles_3():
    """12 cells, some neighbours."""
    box = [20, 15, 20]
    r_cut = 5 
    nb_order = create_nb_order(box, r_cut)

    ppos = np.array([[0, 0, 0], [0, 0, 5], [6, 0, 5], [19, 14, 19]])
    
    nl = NeighborList(box, ppos, r_cut)

    params = np.ones([4,3])
    sigma_c = 1
    potential = potentials(
        ppos, params, sigma_c, nl, nb_order, r_cut, lj=True, coulomb=False)
    pot1 = 4 * ((1 / 5)**12 - (1 / 5)**6) 
    pot2 = 4 * ((1 / 3)**6- (1 / 3)**3)
    potential_ref = pot1 + pot2
    npt.assert_almost_equal(potential, potential_ref)
    




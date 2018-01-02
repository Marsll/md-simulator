import numpy as np
from scipy.special import erfc


def coulomb_pot(r1, r2, q1, q2, sigma, box, r_cut=5):
    """Compute the Coulomb potential between two particles."""
    r12 = np.linalg.norm(pbc(r1 - r2, box))
    potential = 0
    if r12 <= r_cut:
        #Gaussian units!!!!!!!!!!!!!!!!!!!!!!!
        potential = q1 * q2 / r12 * erfc(r12 / (np.sqrt(2) * sigma))
    return potential
    
def coulomb_force(r1, r2, q1, q2, sigma, box, r_cut=5):
    """Compute the Coulomb force between two particles.
    The force acts on particle 1 (r1)."""
    dist = pbc(r1 - r2, box)
    r12 = np.linalg.norm(dist)
    force = 0
    direction = 0
    if r12 <= r_cut:
        #Gaussian units!!!!!!!!!!!!!!!!!!!!!!!
        f1 = erfc(r12 / (np.sqrt(2) * sigma)) / r12 
        f2 = np.sqrt(2 / np.pi) / sigma * np.exp(- r12**2 / (2 * sigma**2))
        force = q1 * q2 / r12 * (f1 + f2)
        direction = dist / r12
    return force * direction
    
def pbc(dist, box):
    #box (x-length, y-length, z-length) --> enumerate -> ((1,x-length),...)
    for i, length in enumerate(box):
        while dist[i] >= 0.5 * length:
            dist[i] -= length
        while dist[i] < -0.5 * length:
            dist[i] += length
    return dist

def test_coulomb_pot():
    pos1 = np.array([0])
    pos2 = np.array([100])
    q1 = 1
    q2 = 1
    sigma = 1
    box = (5000,)
    r_cut = 5000
    potential = coulomb_pot(pos1, pos2, q1, q2, sigma, box, r_cut)
    potential_ref = 0
    np.testing.assert_allclose(potential, potential_ref)

def test_coulomb_force():
    pos1 = np.array([0])
    pos2 = np.array([500])
    q1 = 1
    q2 = -1
    sigma = 1
    box = (5000,)
    r_cut = 5000
    force = coulomb_force(pos1, pos2, q1, q2, sigma, box, r_cut)
    force_ref = 0
    np.testing.assert_allclose(force, force_ref)
    
    
    
    
test_coulomb_pot()
test_coulomb_force()    
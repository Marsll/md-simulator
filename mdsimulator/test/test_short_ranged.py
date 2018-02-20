import numpy as np
from ..short_ranged import potentials
from ..neighbor_list import NeighborList
from ..neighbor_order_pbc import create_nb_order
import matplotlib.pyplot as plt
import numpy.testing as npt
from scipy.special import erfc

"""TO DO"""

def test_potential_shape():
    n = 1000
    box = [50]
    ppos = np.array([[0], [1.]])
    params = np.array([[1, 1, 1],
                       [1, 1, 1]])
    sigma_c = 1
    r_cut = 5
    nl = NeighborList(box, ppos, r_cut)
    nbs = create_nb_order(box, r_cut)
    epots = np.zeros(n)
    epots_ref = np.zeros(n)
    rs = np.zeros(n, dtype=float)
    count = 0
    r = 1
    #print(potentials(np.array([[0],[10]]), params, sigma_c, nl, nbs, r_cut))
    while count < n:
        rs[count] = np.linalg.norm(ppos[0, 0] - ppos[1, 0])
        epots[count] = potentials(ppos, params, sigma_c, nl, nbs, r_cut)
        
        epots_ref[count] = 4 * ((1 / r)**12 - (1 / r)**6) +  1 / r * erfc(r / (np.sqrt(2)))  
        
        count += 1
        r += 10. / n
        ppos[1, 0] = ppos[1, 0] + 10. / n
        
        
    npt.assert_almost_equal(epots, epots_ref, decimal = 3)
    #komsich, dass es kleine Abweichungen gibt... 
    print(sum(epots - epots_ref))
    print(epots - epots_ref)
   
    plt.plot(rs[:100], epots[:100])  
    plt.plot(rs[:100], epots_ref[:100]) 


test_potential_shape()    
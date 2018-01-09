import numpy as np
from short_ranged import potentials
from neighbor_list import NeighborList
from neighbor_order_pbc import create_nb_order
import matplotlib.pyplot as plt
import numpy.testing as npt

"""TO DO"""

def test_potential_shape(n):
    box = [10]
    ppos = np.array([[0], [1]])
    params = np.array([[1, 1, 1],
                       [1, 1, 1]])
    sigma_c = 1
    r_cut = 5
    nl = NeighborList(box, ppos, r_cut)
    nbs = create_nb_order(box, r_cut)
    epots = np.zeros(n)
    rs = np.zeros(n)
    count = 0
    while count < n:
        rs[count] = np.linalg.norm(ppos[0, 0] - ppos[1, 0])
        epots[count] = potentials(ppos, params, sigma_c, nl, nbs, r_cut)
        count += 1
        print((ppos))
        ppos[1, 0] = ppos[1, 0] + 10. / n
        
    plt.plot(rs, epots)    
    
test_potential_shape(10)    
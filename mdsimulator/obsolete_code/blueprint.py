import numpy as np
from mdsimulator.neighbor_list import NeighborList
from mdsimulator.neighbor_order_pbc import create_nb_order
from mdsimulator.short_ranged import potentials

# n number of particles
n = 2
ppos = np.array([[1, 2],
                  [2, 4]])
box = (10, 10)
params = 3
sigma_c = 1
r_cut = 10

nl = NeighborList(box, ppos, r_cut)
nbs = create_nb_order(box, r_cut)

e = potentials(ppos, params, sigma_c, nl, nbs, r_cut, lj=True, couloumb=True)
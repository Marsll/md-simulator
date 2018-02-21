import numpy as np
from mdsimulator import ewald 
from mdsimulator.neighbor_list import NeighborList
from mdsimulator.neighbor_order_pbc import create_nb_order
from mdsimulator.short_ranged import potentials

# n number of particles
n = 2
ppos=np.array([[3,3,3],[3.,3.,4.]])
box=np.array([10,10,10])
params=np.array([np.array([1,-1]),np.array([0.0115897,0.4184]),np.array([3.3284,4.40104]),np.array([22.9898,35.453])]).T

alpha = 5
r_cut = 10
k_max=5


nl = NeighborList(box, ppos, r_cut)
nbs = create_nb_order(box, r_cut)
s = potentials(ppos, params, 1/(np.sqrt(2)*alpha), nl, nbs, r_cut, lj=False, coulomb=True)
l = ewald.longrange(ppos,params.T[0],box,k_max,alpha,potential=True,forces=False) + ewald.self_energy(params.T[0],alpha)

print(s,l)
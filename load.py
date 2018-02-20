import numpy as np
#from mdsimulator import neighbor_list
#from mdsimulator import neighbor_order_pbc
from mdsimulator import mcmc_short_ranged
#from mdsimulator import short_ranged


with np.load('sodium-chloride-example.npz') as fh:
    #dimensions of the box
    box = fh['box']
    #all positions
    positions = fh['positions']
    types = fh['types']
    parameters = fh['parameters'].item()
    

#q, epsilon, sigma, m
params = np.empty([len(positions),4])


for key in parameters:
    params[np.where(types == key)]= parameters[key]

#order of the parameters is shifted from sigma, epsilon, mass, charge to charge, epsilon, sigma, mass 
params[:,[0, 3]] = params[:,[3, 0]]
params[:,[2, 3]] = params[:,[3, 2]]

sigma_c = 1
#cutoff radius
r_cut = 5

finalppos, potential, _a,v = mcmc_short_ranged.mcmc(positions, params, sigma_c, box, r_cut, alpha=0.1, beta=10000, tol=1E-8,
         max_steps=100)
print(potential)
import numpy as np
import matplotlib.pyplot as plt
from mdsimulator.optimize import Optimizer
from mdsimulator import analysis

with np.load('sodium-chloride-example.npz') as fh:
    # dimensions of the box
    box = fh['box']
    # all positions
    positions = fh['positions']
    types = fh['types']
    parameters = fh['parameters'].item()


# q, epsilon, sigma, m
params = np.empty([len(positions), 4])


for key in parameters:
    params[np.where(types == key)] = parameters[key]

# Order of the parameters is shifted from sigma, epsilon, mass, charge
# to charge, epsilon, sigma, mass
params[:, [0, 3]] = params[:, [3, 0]]
params[:, [2, 3]] = params[:, [3, 2]]

# Standard deviation for Gaussian charge distribution in Ewald summation
alpha = 0.2
sigma_c = 1 / (np.sqrt(2) * alpha)
# Cutoff radius
r_cut = 5

opt = Optimizer(box, positions, params, r_cut, sigma_c)
#opt.set_run_options(n_steps=5, storeppos=True)
# opt.run()

print(opt.get_energy())
# epots = opt.get_energies()
# ppos_series = opt.get_ppos_series()
# last_ppos = opt.get_ppos()


# plt.plot(epots)
# analysis.plot_positions(last_ppos)
# plt.show()

# print(params[:,0])

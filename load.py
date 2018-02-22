import numpy as np
import matplotlib.pyplot as plt
from mdsimulator.optimize import Optimizer
from mdsimulator import analysis
from mdsimulator import rdf

#########################################################################
# Load data from npz file
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

###########################################################################
# Specify important parameters for the calculations

# Standard deviation for Gaussian charge distribution in Ewald summation
# sigma_c = 1 / (np.sqrt(2) * alpha)
alpha = 0.2

# Cutoff radius
r_cut = 5

# Maximum k space vectors taken into account
k_max = 10

############################################################################
# Specify options for the Markov Chain optimization

# Number of steps in the chain
n_steps = 50

# beta = 1/(kB * T)
# high beta - low temperature - small chance to accept if energy higher
# low beta - high temperature - high chance to accept even if energy higher
temperature = 20

# Scaling factor for displacement of each particle in one Markov chain step
step_width = 0.1

# Want to save the entire series of ppos arrays?
storeppos = True
############################################################################
# Initialize the optimizer with all given parameters and data

opt = Optimizer(box, positions, params, r_cut, alpha, k_max)
opt.set_run_options(n_steps=n_steps, temperature=temperature,
                    step_width=step_width, storeppos=storeppos)

############################################################################
# Run the optimization and obtain energies and positions
opt.run()
#histo, bins = rdf.rdf(np.asarray(opt.get_ppos_series()), box)

# plt.figure()
# plt.title("Radial distribution function")
# plt.plot(bins, histo)

epots = opt.get_total_energies()
e_shorts = opt.get_short_energies()
e_longs = opt.get_long_energies()
e_selfs = np.zeros(n_steps) + opt.get_energy_self()
ppos_series = opt.get_ppos_series()
last_ppos = opt.get_ppos()

analysis.plot_energies((epots, "total"), (e_longs, "long"),
                       (e_shorts, "short"), (e_selfs, "self"))

analysis.plot_positions(last_ppos, params[:, 0])



######################################################################
# Set up charges on a perfect grid and optimize
# Noise parameter displaces them from the perfect grid

# noise = 1.0
# ppos_grid = analysis.grid_positions(types, box, noise)

# opt_grid = Optimizer(box, ppos_grid, params, r_cut, alpha, k_max)
# opt_grid.set_run_options(n_steps=500, temperature=300,
#                          step_width=0.3, storeppos=storeppos)
# opt_grid.run()

# epots_grid = opt_grid.get_total_energies()
# e_shorts_grid = opt_grid.get_short_energies()
# e_longs_grid = opt_grid.get_long_energies()
# e_selfs_grid = np.zeros(len(epots_grid)) + opt.get_energy_self()
# ppos_series_grid = opt_grid.get_ppos_series()
# last_ppos_grid = opt_grid.get_ppos()


# analysis.plot_energies((epots_grid, "total"), (e_longs_grid, "long"),
#                        (e_shorts_grid, "short"), (e_selfs_grid, "self"))

# plt.figure()
# plt.title("Distribution of energies")
# plt.hist(epots_grid, bins=30)

# analysis.plot_positions(last_ppos_grid, params[:, 0])




plt.show()

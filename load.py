import numpy as np
import matplotlib.pyplot as plt
from mdsimulator.optimize import Optimizer
from mdsimulator import analysis

#########################################################################
# Load data from npz file
with np.load('sodium-chloride-example.npz') as fh:
    # dimensions of the box
    box = fh['box']
    # all positions
    positions = fh['positions']
    types = fh['types']
    parameters = fh['parameters'].item()


#new grid test positions, which should be nearly optimal
x = np.linspace(0, box[0], 4, endpoint=False)
xx, yy, zz = np.meshgrid(x,x,x)
positions1 = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

x = np.linspace(0 + box[0] / 8, box[0] + box[0] / 8, 4, endpoint=False)
xx, yy, zz = np.meshgrid(x,x,x)
positions2 = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
positions = np.concatenate((positions1, positions2), axis=0)


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
n_steps = 1000

# beta = 1/(kB * T)
# high beta - low temperature - small chance to accept if energy higher
# low beta - high temperature - high chance to accept even if energy higher
beta = 100

# Scaling factor for displacement of each particle in one Markov chain step
step_width = 0.1

# Want to save the entire series of ppos arrays?
storeppos = True
############################################################################
# Initialize the optimizer with all given parameters and data

opt = Optimizer(box, positions, params, r_cut, alpha, k_max)
opt.set_run_options(n_steps=n_steps, beta=beta,
                    step_width=step_width, storeppos=storeppos)

############################################################################
# Run the optimization and obtain energies and positions
opt.run()


epots = opt.get_energies()
ppos_series = opt.get_ppos_series()
last_ppos = opt.get_ppos()


plt.plot(epots)
analysis.plot_positions(last_ppos, params[:, 0])
plt.show()


print("total", opt.get_energy())
print("long", opt.get_energy_long())
print("short", opt.get_energy_short())
print("self", opt.get_energy_self())

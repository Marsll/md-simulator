import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np


def plot_positions(ppos, charges):
    fig = plt.figure()
    dims = ppos.shape[1]
    colorsequence = np.empty(len(ppos), dtype=str)

    colors = ["red", "blue", "deepskyblue", "magenta"]
    for i, value in enumerate(np.unique(charges)):
        colorsequence[charges == int(value)] = colors[i]

    if dims == 3:
        ax = fig.gca(projection='3d')
        ax.scatter(*ppos.T, marker="o", color=colorsequence)
    elif dims == 2:
        plt.scatter(*ppos.T, marker="o", color=colorsequence)
    elif dims == 1:
        y = np.zeros(ppos.shape[0])
        plt.plot(*ppos.T, np.zeros_like(y), "o", color=colorsequence)


def plot_forces(ppos, forces):
    fig = plt.figure()
    dims = ppos.shape[1]

    if dims == 3:
        ax = fig.gca(projection='3d')
        ax.quiver(*ppos.T, *forces.T, length=0.1, normalize=True)
    elif dims == 2:
        plt.quiver(*ppos.T, *forces.T)
    elif dims == 1:
        plt.quiver(*ppos.T, *forces.T)

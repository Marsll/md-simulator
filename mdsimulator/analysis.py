import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from .metropolis import back_map


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


def grid_positions(types, box, noise):
    d = len(box)
    length = min(box)
    ty = np.unique(types)
    number1 = len(types[types == ty[0]])
    number2 = len(types[types == ty[1]])
    n1 = np.ceil(number1**(1 / d))
    n2 = np.ceil(number2**(1 / d))

    x = np.linspace(0, length, n1, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x)
    positions1 = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

    x = np.linspace(length / 2 / n2, length + length / 2 / n2,
                    n2, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x)
    positions2 = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
    positions = np.concatenate((positions1, positions2), axis=0)
    displacement = noise * np.random.rand(*positions.shape)
    return back_map(positions + displacement, box)

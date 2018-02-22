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


def plot_energies(*args):
    plt.figure()
    ax = plt.subplot(111)
    ax.set_title("Energy over course of Markov chain", y=1.05)
    for arg in args:
        if type(arg) == tuple:
            ax.plot(arg[0], label=arg[1])
        else:
            raise Exception(
                "Arguments should be tuple, where the first index corresponds to an ndarray of energies and the second to a string labeling which part of the energy")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)


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

def plot_rdfs(hist1, hist2, bins, element1, element2):
    plt.figure()
    plt.title("Radial distribution function")
    plt.plot(bins, hist1, label=str(element1))
    plt.plot(bins, hist2, label=str(element2))
    plt.legend()
def test_mcmc():
    """Particles in a periodic box."""
    #ppos = np.random.random([3, 3]) * 5
    ppos = np.array([[0, 0, 0], [2, 0, 0], [3, 2, 1]])
    params = np.ones(ppos.shape)
    params[1, 0] = 1
    sigma_c = 1
    plot_positions(ppos)
    dim_box = (5, 5, 5)
    finalppos, potential, _ = mcmc(ppos, params, sigma_c, dim_box, r_cut=5)
    plot_positions(finalppos)
    print(potential, np.linalg.norm(pbc(finalppos[0] - finalppos[1], dim_box)),
          np.linalg.norm(pbc(finalppos[2] - finalppos[1], dim_box)),
          np.linalg.norm(pbc(finalppos[0] - finalppos[2], dim_box)))


def mcmc_sampling():
    """Particles in a periodic box."""
    N = 64
    ppos = np.random.random([N, 3]) * 10
    # plot_positions(ppos)
    dim_box = (10, 10, 10)
    params = np.ones(ppos.shape)
    params[: int(N / 2), 0] = -1
    # print(params)
    beta = 100000
    sigma_c = 1
    finalppos, potential, epots, ppos_array = mcmc(ppos, params, sigma_c, dim_box, r_cut=3,
                                                   step_width=.5,
                                                   beta=beta, max_steps=100)

    plt.figure()
    histo_average, bins = rdf(ppos_array, dim_box)
    print(histo_average)
    plt.plot(bins, histo_average)
    plt.show()

    return epots


def boltzmann_distribution(e_min, e_max, beta, N):
    e_arr = np.linspace(e_min, e_max, 10000)
    n_arr = N * np.exp(-beta * e_arr**2)
    return e_arr, n_arr
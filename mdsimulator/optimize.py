from .neighbor_order_pbc import create_nb_order
from .neighbor_list import NeighborList
from .short_ranged import potentials
from .ewald import longrange, self_energy
from .metropolis import mcmc_step


class Optimizer:

    def __init__(self, box, ppos, params, r_cut, alpha, k_max):
        self.get_system(box, ppos, params, r_cut, alpha, k_max)
        self.get_nb_order()
        self.get_neighbor_list()
        self.calc_energy()

        self.epots = [self.e]
        self.ppos_arr = [self.ppos]

        self.run_options = {
            "n_steps": 100,
            "beta": 1000,
            "step_width": 0.1,
            "storeppos": False
        }

    def get_system(self, box, ppos, params, r_cut, alpha, k_max):
        self.box = box
        self.params = params
        self.ppos = ppos
        self.alpha = alpha
        self.sigma_c = 1 / (2**(0.5) * alpha)
        self.r_cut = r_cut
        self.k_max = k_max

    def get_nb_order(self):
        self.nbs = create_nb_order(self.box, self.r_cut)

    def get_neighbor_list(self):
        self.nl = NeighborList(self.box, self.ppos, self.r_cut)

    def calc_energy(self):
        self.e_short = potentials(self.ppos, self.params, self.sigma_c,
                                  self.nl, self.nbs, self.r_cut)

        self.e_long = longrange(self.ppos, self.params[:, 0], self.box,
                                self.k_max, self.alpha, potential=True,
                                forces=False)

        self.e_self = self_energy(self.params[:, 0], self.alpha)

        self.e = self.e_short + self.e_long + self.e_self

    def set_run_options(self, **kwargs):
        for key in kwargs.keys():
            if key in self.run_options:
                self.run_options[key] = kwargs[key]
            else:
                raise Exception("Unexpected argument")

    def run(self):
        for _ in range(self.run_options["n_steps"]):
            self.ppos, self.e = mcmc_step(
                self.ppos, self.params, self.sigma_c, self.box,
                self.r_cut, self.alpha, self.k_max, self.nbs, self.nl, self.e,
                self.run_options["step_width"], self.run_options["beta"])

            self.epots.append(self.e)

            if self.run_options["storeppos"]:
                self.ppos_arr.append(self.ppos)

    def get_energy(self):
        return self.e

    def get_ppos(self):
        return self.ppos

    def get_energies(self):
        return self.epots

    def get_ppos_series(self):
        return self.ppos

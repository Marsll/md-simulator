from .neighbor_order_pbc import create_nb_order
from .neighbor_list import NeighborList
from .short_ranged import potentials
from .ewald import longrange, self_energy
from .metropolis import mcmc_step
import scipy.constants


class Optimizer:

    def __init__(self, box, ppos, params, r_cut, alpha, k_max):
        """
        Arguments:
            box         (ndarray):      A one dimensional numpy-array with d elements (size of preriodic box)
            ppos        (ndarray):      A two-dimensional array with shape (n,d) (Positions of all particles)   
            params      (ndarray):      A two-dimensional array with shape (n,4) (charge, epsillon, sigma, mass) prameters of all paricles 
            r_cut       (float):        A positive float (cutoff radius in real space)
            alpha       (float):        A positive float (standard deviation for Gaussian charge distribution in Ewald summation)
            k_max       (float):        A positive float (cutoff radius in k space)
        """
        self.get_system(box, ppos, params, r_cut, alpha, k_max)
        self.get_nb_order()
        self.get_neighbor_list()
        self.calc_energy()

        self.epots = [self.e]
        self.e_shorts = [self.e_short]
        self.e_longs = [self.e_long]
        self.ppos_arr = [self.ppos]

        self.run_options = {
            "n_steps": 100,
            "temperature": 300,
            "step_width": 0.1,
            "storeppos": False
        }

        self.set_beta()

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
        self.set_beta()

    def set_beta(self):    
        na = scipy.constants.Avogadro
        kB = scipy.constants.Boltzmann
        self.beta = 1000 / na / (self.run_options["temperature"] * kB)

    def run(self):
        for _ in range(self.run_options["n_steps"]):
            self.ppos, self.e_short, self.e_long = mcmc_step(
                self.ppos, self.params, self.sigma_c, self.box,
                self.r_cut, self.alpha, self.k_max, self.nbs, self.nl,
                self.e_short, self.e_long,
                self.run_options["step_width"], self.beta)

            self.e_shorts.append(self.e_short)
            self.e_longs.append(self.e_long)

            self.e = self.e_short + self.e_long + self.e_self
            self.epots.append(self.e)

            if self.run_options["storeppos"]:
                self.ppos_arr.append(self.ppos)

    def get_energy(self):
        return self.e

    def get_energy_short(self):
        return self.e_short

    def get_energy_long(self):
        return self.e_long

    def get_energy_self(self):
        return self.e_self

    def get_ppos(self):
        return self.ppos

    def get_total_energies(self):
        return self.epots

    def get_short_energies(self):
        return self.e_shorts

    def get_long_energies(self):
        return self.e_longs

    def get_ppos_series(self):
        return self.ppos_arr

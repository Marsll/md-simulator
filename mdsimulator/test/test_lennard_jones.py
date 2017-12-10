from ..lennard_jones import lenard_jones_forces
import numpy as np


np.testing.assert_array_equal(lenard_jones_forces(0, 1, epsillon = 1, sigma = 1)[1], 2)


import numpy as np
from StateVector import StateVector, StateVectorVar
from DifferenceEqs import calc_g

class TestDifferenceEqs:
    def test_calc_g(self):
        sv = StateVector(10, True)
        params = {
            "mu": 1,
            "E0_prime": 2,
            "k0_prime": 3
        }
        output = calc_g(sv, params)
        assert (output.shape==(9*4,))

import numpy as np
import pytest
import Minimizer
import Utilities
import Integrator

class TestMinimizer:
    def test_minimizer(self):
        # Make sure that When using Sun parameters, you get core temps on the same order as the accepted values
        length_scale = Utilities.R_sun
        mass_scale = Utilities.M_sun
        mu = Utilities.mu_sun
        scale_factors = Utilities.UnitScalingFactors(mass_scale, length_scale)
        params = Utilities.generate_extra_parameters(mass_scale, length_scale,Utilities.E_0_sun, Utilities.kappa_0_sun,Utilities.mu_sun)
        results = Minimizer.run_minimizer(2E7/scale_factors[Utilities.TEMP_UNIT_INDEX],30E15/scale_factors[Utilities.PRESSURE_UNIT_INDEX],
                                        1000, mass_scale, length_scale, params["E_prime"], params["kappa_prime"], mu)
        estimated_core_pressure = results.x[1]*scale_factors[Utilities.PRESSURE_UNIT_INDEX]
        estimated_core_temp = results.x[0]*scale_factors[Utilities.TEMP_UNIT_INDEX]
        expected_core_temp = 1.5E7 # K
        expected_core_pressure = 26.5E6*1E9 # Pa
# This one is off by a factor of 2 for some reason
        assert(np.floor(np.log10(estimated_core_temp))==np.floor(np.log10(expected_core_temp)))
# This one is off by a factor of 3 for some reason
        assert(np.floor(np.log10(estimated_core_pressure))==np.floor(np.log10(expected_core_pressure)))
if __name__ == "__main__":
    pass
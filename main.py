import numpy as np
import matplotlib.pyplot as plt
import Minimizer
import Integrator
import Utilities

if __name__ == "__main__":
    length_scale = Utilities.R_sun
    mass_scale = Utilities.M_sun
    mu = Utilities.mu_sun
    scale_factors = Utilities.UnitScalingFactors(mass_scale, length_scale)
    params = Utilities.generate_extra_parameters(mass_scale, length_scale,0.1,0.02,Utilities.mu_sun)
    print("Scale: ", scale_factors)
    print("Extra: ", params)
    init = Minimizer.gen_initial_conditions(2E7/scale_factors[Utilities.TEMP_UNIT_INDEX],30E15/scale_factors[Utilities.PRESSURE_UNIT_INDEX], 1/1000, params)
    mesh = Integrator.ODESolver(init,1000,params)
#    results = Minimizer.run_minimizer(30E15/scale_factors[Utilities.PRESSURE_UNIT_INDEX] , 2E7/scale_factors[Utilities.TEMP_UNIT_INDEX],
#                                       1000, mass_scale, length_scale, params["E_prime"], params["kappa_prime"], mu)
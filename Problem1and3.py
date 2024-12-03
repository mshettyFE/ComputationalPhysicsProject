import numpy as np
import matplotlib.pyplot as plt
import Minimizer
import Integrator
from Utilities import *

if __name__ == "__main__":
    # Run the sun test
    length_scale = R_sun
    mass_scale = M_sun
    mu = mu_sun
    scale_factors = UnitScalingFactors(mass_scale, length_scale)
    constants = generate_extra_parameters(mass_scale, length_scale, E_0_sun, kappa_0_sun, mu_sun)
    print("Extra:", constants)
    results = Minimizer.run_minimizer(2E7/scale_factors[TEMP_UNIT_INDEX],30E15/scale_factors[PRESSURE_UNIT_INDEX],
                                       1000, mass_scale, length_scale, constants["E_prime"], constants["kappa_prime"], mu)
    print("res:", results)
    init_conds = Minimizer.gen_initial_conditions(results.x[0], results.x[1],1E-2, constants)
    print("Initial", init_conds)
    state0 = Integrator.ODESolver(init_conds, 1000, constants)
#    print(state0.shape)
#    print(state0)
    np.savetxt("SunMesh.txt", state0,delimiter=",")
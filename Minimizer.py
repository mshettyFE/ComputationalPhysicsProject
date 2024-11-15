import numpy as np
import scipy as sp
import Integrator
import Utilities

def loss_function(estimator_guess, *args):
    """
        Input:
            estimator_guess: 2x1 numpy array of the form [temp, pressure]
            *args: Expect two additional args: the ODE solver, and the number of steps the ODE solver should take
        Output:
            np.float64: the loss function for the given initial pressure and temp conditions
    """
# We assume that estimator has dimensions 2x1
# Temperature and pressure normally can't be negative
    initial_pressure  = estimator_guess[0]
    initial_temp  = estimator_guess[1]
    assert(initial_pressure >= 0) 
    assert(initial_temp >= 0)
# Want temperature and pressure to be 0 at boundaries. These variables are mostly just for clarity
    expected_pressure = np.float64(0)
    expected_temp = np.float64(0)
    expected_final_mass = 1
# *args should of the form  [ODE solver,n_steps, extra_const_params]
    assert(len(args) == 3)
    solver, n_steps, const_params = args
# encode the boundary conditions of m'= L' = r'=0, plug put in the initial temp and pressure guesses
    initial_conds = np.array(  (0,0, Utilities.equation_of_state(initial_pressure, initial_temp,const_params),
                                initial_pressure , 0,initial_temp) )
# use the ODE solver to propagate the initial conditions to the final state
# This signature might be wrong though since it hasn't been developed yet
    time_evolution = solver(initial_conds, n_steps, const_params)
    final_mass = time_evolution[Utilities.MASS_UNIT_INDEX,-1]
    final_pressure = time_evolution[Utilities.PRESSURE_UNIT_INDEX,-1]
    final_temp = time_evolution[Utilities.TEMP_UNIT_INDEX,-1]
# Make sure that mass is 1
    assert( (final_mass - expected_final_mass) < Utilities.global_tolerance)
    return   np.pow((final_pressure- expected_pressure),2) + np.pow(final_temp- expected_temp,2)

def run_minimizer(Initial_T, Initial_P, num_iters, R_0, M_0, epsilon, kappa, mu):
    x0 = np.array([Initial_T, Initial_P])
    solver = Integrator.ODESolver
    extra_const_params = Utilities.generate_extra_parameters(R_0, M_0, epsilon, kappa, mu)
    return sp.optimize.minimize(loss_function,x0, args=(solver,num_iters,extra_const_params))

if __name__ == "__main__":
    pass

import numpy as np
import scipy as sp
import Integrator
import Utilities

#MASS_UNIT_INDEX = 0 []
#RADIUS_UNIT_INDEX = 1 []
#DENSITY_UNIT_INDEX = 2 []
#PRESSURE_UNIT_INDEX = 3 [dynes/cm^2]
#LUMINOSITY_UNIT_INDEX = 4 []
#TEMP_UNIT_INDEX = 5 [K]
#TIME_UNIT_INDEX = 6 

def gen_initial_conditions(starting_scaled_temp, starting_scaled_pressure, step_size, const_params):
    """
        Helper function to deal with the fact that we can't start at m=0. We pretend that the central density is roughly constant, then fudge the boundary conditions a bit
        starting_scaled_temp: the initial scaled temperature at the center of the star (unitless)
        starting_scaled_pressure: same as temp, but for pressure
        step_size: the mass step size to be taken (typically, this should be half a step size of your actual simulation)
        const_params: dictionary containing the constant parameters of the problem. Generated from Utilities.generate_extra_parameters
    """
    initial_density =  Utilities.equation_of_state(starting_scaled_pressure, starting_scaled_temp,const_params)
# encode the boundary conditions of m'= L' = r'=0, plug put in the initial temp and pressure guesses
# We need to fudge the radius initial condition to avoid the singularity at r=0 in the equations.
    initial_mass = step_size/2
    initial_rad = np.power((4*np.pi/3)*initial_mass/initial_density, 1/3)
    initial_conds = np.array(  (initial_mass,initial_rad,initial_density,
                                starting_scaled_pressure , 0, starting_scaled_temp) )
    return initial_conds

def loss_function(estimator_guess, *args):
    """
        Input:
            estimator_guess: 2x1 numpy array of the form [temp, pressure]. These should be unitless
            *args: Expect two additional args: the ODE solver, and the number of steps the ODE solver should take
        Output:
            np.float64: the loss function for the given initial pressure and temp conditions
    """
# We assume that estimator has dimensions 2x1
# Temperature and pressure normally can't be negative
    initial_pressure  = estimator_guess[0]
    initial_temp  = estimator_guess[1]
#    assert(initial_pressure >= 0) 
#    assert(initial_temp >= 0)
# Want temperature and pressure to be 0 at boundaries. These variables are mostly just for clarity
    expected_pressure = np.float64(0)
    expected_temp = np.float64(0)
# *args should of the form  [ODE solver,n_steps, extra_const_params]
    assert(len(args) == 3)
    solver, n_steps, const_params = args
    initial_conds = gen_initial_conditions(initial_temp, initial_pressure, 1/n_steps, const_params)
# use the ODE solver to propagate the initial conditions to the final state
    time_evolution = solver(initial_conds, n_steps, const_params)
    print(time_evolution[Utilities.TEMP_UNIT_INDEX,:10])
    final_mass = time_evolution[Utilities.MASS_UNIT_INDEX,-1]
    final_pressure = time_evolution[Utilities.PRESSURE_UNIT_INDEX,-1]
    final_temp = time_evolution[Utilities.TEMP_UNIT_INDEX,-1]
    return   np.pow((final_pressure- expected_pressure),2) + np.pow(final_temp- expected_temp,2)

def run_minimizer(Initial_scaled_T, Initial_scaled_P, num_iters, M_0, R_0, epsilon, kappa, mu):
    """
        Helper function: to generate set up the minimizer and run it
        Initial_scaled_T: Initial guess of temperature (unitless)
        Initial_scaled_P: Initial guess of pressure (unitless)
        num_iters: how many steps the integrator should take (int >0)
        M_0: the relevant mass scale of the problem (kg)
        R_0: The relevant distance scale of the problem (m)
        epsilon: the e_0 parameter in the luminosity differential equation
        kappa: the k_0 parameter in the temperature differential equation
        mu: the mean molecular weight in units of proton mass
    """
    x0 = np.array([Initial_scaled_T, Initial_scaled_P])
    solver = Integrator.ODESolver
    extra_const_params = Utilities.generate_extra_parameters(M_0, R_0, epsilon, kappa, mu)
    return sp.optimize.minimize(loss_function,x0,
                                args=(solver,num_iters,extra_const_params),
                                bounds=sp.optimize.Bounds( # Hopefully, prevent Minimizer from guessing a negative temperature...
                                    lb=[Utilities.global_tolerance,Utilities.global_tolerance],
                                    ub=[np.inf,np.inf], keep_feasible=[True,True]),
                                )

if __name__ == "__main__":
    pass

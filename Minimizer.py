import numpy as np
import scipy as sp
import Integrator
import Utilities

# Minimum bound on tolea
bound_tol = 1E-9

def gen_initial_conditions(starting_scaled_temp, starting_scaled_pressure, step_size, const_params):
    """
        Input:
            Helper function to deal with the fact that we can't start at m=0. We pretend that the central density is roughly constant, then fudge the boundary conditions a bit
            starting_scaled_temp: the initial scaled temperature at the center of the star (unitless)
            starting_scaled_pressure: same as temp, but for pressure
            step_size: the mass step size to be taken (typically, this should be half a step size of your actual simulation)
            const_params: dictionary containing the constant parameters of the problem. Generated from Utilities.generate_extra_parameters
        Output:
            1x6 numpy array containing initial conditions in scaled variables
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
#    print(estimator_guess)
    initial_pressure  = estimator_guess[0]
    initial_temp  = estimator_guess[1]
# Want temperature and pressure to be 0 at boundaries. These variables are mostly just for clarity
    expected_pressure = np.float64(0)
    expected_temp = np.float64(0)
# *args should of the form  [ODE solver,n_steps, extra_const_params]
    assert(len(args) == 3)
    solver, n_steps, const_params = args
    initial_conds = gen_initial_conditions(initial_temp, initial_pressure, 1/n_steps, const_params)
# use the ODE solver to propagate the initial conditions to the final state
    time_evolution = solver(initial_conds, n_steps, const_params)
    final_pressure = time_evolution[-1,Utilities.PRESSURE_UNIT_INDEX]
    final_temp = time_evolution[-1,Utilities.TEMP_UNIT_INDEX]
# Create concave function to minimize
    return   np.pow((final_pressure- expected_pressure),2) + np.pow(final_temp- expected_temp,2)

def run_minimizer(Initial_scaled_T, Initial_scaled_P, num_iters, M_0, R_0, epsilon, kappa, mu):
    """
        Helper function: to generate set up the minimizer and run it
            Input:
                Initial_scaled_T: Initial guess of temperature (unitless)
            Initial_scaled_P: Initial guess of pressure (unitless)
            num_iters: how many steps the integrator should take (int >0)
            M_0: the relevant mass scale of the problem (kg)
            R_0: The relevant distance scale of the problem (m)
            epsilon: the e_0 parameter in the luminosity differential equation
            kappa: the k_0 parameter in the temperature differential equation
            mu: the mean molecular weight in units of proton mass
        Output:
            OptimizeResult from scipy.optimize.minimize
    """
    x0 = np.array([Initial_scaled_T, Initial_scaled_P])
    solver = Integrator.ODESolver
    extra_const_params = Utilities.generate_extra_parameters(M_0, R_0, epsilon, kappa, mu)
    return sp.optimize.minimize(loss_function,x0,
                                args=(solver,num_iters,extra_const_params),
                                method =  "L-BFGS-B",
                                bounds=sp.optimize.Bounds( # Hopefully, prevent Minimizer from guessing a negative temperature...
                                    lb=[bound_tol,bound_tol],
                                    ub=[np.inf,np.inf], keep_feasible=[True,True]),
                                )

def grid_search(Initial_scaled_T, Initial_scaled_P, num_iters, M_0, R_0,mu, grid_size=10, verbose=False):
    """
        Helper function: to perform a grid search in epsilon and kappa to find the correct order of magnitude for both
            Initial_scaled_T: Initial guess of temperature (unitless)
            Initial_scaled_P: Initial guess of pressure (unitless)
            num_iters: how many steps the integrator should take (int >0)
            M_0: the relevant mass scale of the problem (kg)
            R_0: The relevant distance scale of the problem (m)
            mu: the mean molecular weight in units of proton mass
            grid_size: Number of orders of magnitude to crawl through in one direction
            verbose: debugging variable
        Output:
            OptimizeResult from scipy.optimize.minimize
    """
    epsilon_scaling = np.arange(-grid_size,grid_size+1)
    kappa_scaling = np.arange(-grid_size,grid_size+1)
    x0 = np.array([Initial_scaled_T, Initial_scaled_P])
    solver = Integrator.ODESolver
    output = []
    for e_i, e in enumerate(epsilon_scaling):
        for k_i, k in enumerate(kappa_scaling):
            constants = Utilities.generate_extra_parameters(M_0, R_0, e*Utilities.E_0_sun, k*Utilities.kappa_0_sun, mu)
            results = sp.optimize.minimize(loss_function,x0,
                                args=(solver,num_iters,constants),
                                bounds=sp.optimize.Bounds(
                                    lb=[bound_tol,bound_tol],
                                    ub=[np.inf,np.inf], keep_feasible=[True,True]),
                                )
            init_conds = gen_initial_conditions(results.x[0], results.x[1],1E-2, constants)
            state0 = Integrator.ODESolver(init_conds, 1000, constants, verbose=True)  
            if (verbose):
                print(e_i,e, k_i, k, state0.shape)
            # If the integrator made it past the first step, save the grid point and output
            if(state0.shape[0] != 1):
                output.append((e,k, state0))
    return output  

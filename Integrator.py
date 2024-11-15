import numpy as np
import scipy as sp;
import Utilities

# Derivative of the dependent variables
# Notably, density is absent. Use Utilities.equation_of_state to generate the density

def r_prime_der(current, extra_const_params):
    new_r = (1/(4*np.pi))*(1/(np.power(current[Utilities.RADIUS_UNIT_INDEX],2)*current[Utilities.DENSITY_UNIT_INDEX]))
    output = current
    output[Utilities.RADIUS_UNIT_INDEX] = new_r
    return output

def P_prime_der(current, extra_const_params):
    new_p = (-1/(4*np.pi))* (current[Utilities.MASS_UNIT_INDEX])/(np.power(current[Utilities.RADIUS_UNIT_INDEX], 4) )
    output = current
    output[Utilities.PRESSURE_UNIT_INDEX] = new_p
    return output

def L_prime_der(current, extra_const_params):
    # assume that epsilon_prime is in args
    new_L = extra_const_params["epsilon_prime"]* current[Utilities.DENSITY_UNIT_INDEX]*np.power(current[Utilities.TEMP_UNIT_INDEX],4)
    output = current
    output[Utilities.LUMINOSITY_UNIT_INDEX] = new_L
    return output

def T_prime_der(current, extra_const_params):
    new_T = - extra_const_params["kappa_prime"]* current[Utilities.DENSITY_UNIT_INDEX] * current[Utilities.LUMINOSITY_UNIT_INDEX] * np.power(current[Utilities.RADIUS_UNIT_INDEX],-4) * np.power(current[Utilities.TEMP_UNIT_INDEX],-6.5)

def RK4(f,current, step_size):
    """
    Straightforward implementation of RK4 algorithm
    Inputs:
        f: derivative of dependent variable. Takes form f(current, extra_const_params),
            * current encodes the current state of the system as a 6x1 np array
            * constant_dict is a Python dictionary which holds any constants needed amongst all the diff eqs
            f should output a numpy array with size of current (6x1)
        y_n: numpy array which holds the current state of the system
            0th term is independent variable, and the rest are the dependent ones
        step_size: how big of a step in x do you want
    """
    assert(step_size >0)
#    assert(current.shape == (6,))
    step_size = np.float64(step_size)
    half_step_size = step_size/2
    
    k1 = f(current)
    new_input =current+half_step_size*k1 
    new_input[0] = current[0]+half_step_size

    k2 = f(new_input)
    
    new_input =current+half_step_size*k2 
    new_input[0] = current[0]+half_step_size

    k3 = f(new_input)
    k3[0] = 1 

    new_input = current+step_size*k3
    new_input[0] = current[0]+step_size
    
    k4 = f(new_input)
    update = (step_size/6) * (k1+2*k2+2*k3+k4)

    # Return updated values, forcing x coordinate to be x_old+step_size
    next_state = current+ update 
    next_state[0] = current[0]+step_size
    return next_state

if __name__ == "__main__":
    pass

def ODESolver(intial_conditions, num_steps, extra_const_parameters):
    """
        Inputs:
            initial_conditions (6x1 np array): The initial conditions of the system
            num_steps (np.float64): The number of steps to take between 0 and 1
        Outputs:
            state_matrix ( (6xnum_steps) np array):
                The state of the system at each mass step in time. Needed for plotting reasons
                The final state should be given by state_matrix[:,-1]
    """
    delta_m = 1/num_steps
    derivative_functions = [r_prime_der, P_prime_der, L_prime_der, T_prime_der]
    assert(intial_conditions.shape[0] ==6)
    assert(intial_conditions[0] == 0)
    output = np.zeros((6,num_steps))
# Do the rest of the ODE solver here
# The indexing I provided you earlier is incorrect, since we need to include \rho in the state vector. My mistake...
# Related to the above: Don't forget to use the equation of state to update the density as well after doing all of the 
    return output

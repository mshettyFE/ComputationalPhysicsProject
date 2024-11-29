import numpy as np
import scipy as sp;
from Utilities import *

# Derivatives of the dependent variables.
# Notably, density is absent. Use Utilities.equation_of_state to generate the density.
def r_prime_der(current,extra_const_parameters):
    new_r = (1/(4*np.pi))*(1/(np.power(current[RADIUS_UNIT_INDEX],2)*current[DENSITY_UNIT_INDEX]))
    output = np.zeros(current.shape)
    output[RADIUS_UNIT_INDEX] = new_r
    return output

def P_prime_der(current,extra_const_parameters):
    new_P = (-1/(4*np.pi))* (current[MASS_UNIT_INDEX])/(np.power(current[RADIUS_UNIT_INDEX], 4) )
    output = np.zeros(current.shape)
    output[PRESSURE_UNIT_INDEX] = new_P
    return output

def L_prime_der(current, extra_const_params):
    # assume that E_prime is in args
    new_L = extra_const_params["E_prime"]* current[DENSITY_UNIT_INDEX]*np.power(current[TEMP_UNIT_INDEX],4)
    output = np.zeros(current.shape)
    output[LUMINOSITY_UNIT_INDEX] = new_L
    return output

def T_prime_der(current, extra_const_params):
    # assume that kappa_prime is in args
    output = np.zeros(current.shape)
    multiplied_vars = current[TEMP_UNIT_INDEX]* current[RADIUS_UNIT_INDEX]
    var = np.power(multiplied_vars, -4)
    tp = np.power(current[TEMP_UNIT_INDEX],-2.5)
    new_T = - extra_const_params["kappa_prime"]* current[DENSITY_UNIT_INDEX] * current[LUMINOSITY_UNIT_INDEX] * var * tp 
    output[TEMP_UNIT_INDEX] = new_T
    return output

def RK4(f, current, step_size, extra_const_params):
    """
    Straightforward implementation of RK4 algorithm
    Inputs:
        f: derivative of dependent variable. Takes form f(current, extra_const_params),
            * current encodes the current state of the system as a 6x1 np array
            * constant_dict is a Python dictionary which holds any constants needed amongst all the diff eqs
            f should output a numpy array with size of current (6x1)
        current: numpy array which holds the current state of the system
            0th term is independent variable, and the rest are the dependent ones
        step_size: how big of a step in x do you want
    """
    assert(step_size >0)
#   assert(current.shape == (6,))
    step_size = np.float64(step_size)
    half_step_size = step_size/2
    
    #k1-k4 are reference points for RK integration.
    k1 = f(current,extra_const_params)
    new_input = current + half_step_size*k1 
    new_input[0] = current[0] + half_step_size

    k2 = f(new_input,extra_const_params)
    
    new_input =current + half_step_size*k2 
    new_input[0] = current[0] + half_step_size
    k3 = f(new_input,extra_const_params)

    new_input = current+step_size*k3
    new_input[0] = current[0]+step_size
    
    k4 = f(new_input,extra_const_params)
    update = (step_size/6) * (k1+2*k2+2*k3+k4)

    # Return updated values, forcing x coordinate to be x_old+step_size
    next_state = current + update 
    next_state[0] = current[0]+step_size
    return next_state

if __name__ == "__main__":
    pass


#Iterates state of system thru RK4, creating an array of the key variables at each mass step.
def ODESolver(initial_conditions, num_steps, extra_const_parameters):
    """
        Inputs:
            initial_conditions (6x1 np array): The initial conditions of the system
            num_steps (np.float64): The number of steps to take between 0 and 1
        Outputs:
            state_matrix ( (6xnum_steps) np array):
                The state of the system at each mass step in time. Needed for plotting reasons
                The final state is given by state_matrix[:,-1]
    """
    step_size = 1/num_steps
    derivatives = [r_prime_der, P_prime_der, L_prime_der, T_prime_der] #array of differential equations.
    state = np.zeros((6,num_steps)) #initializing array of variables.
    state[:,0] = initial_conditions
#    print(initial_conditions[TEMP_UNIT_INDEX], initial_conditions[PRESSURE_UNIT_INDEX])
#    print(initial_conditions)
    for i in range(1, num_steps):
        #RK4 receives a specific differential equation corresponding to each variable of interest. 
        #RK4 outputs a 6x1 array with elements of: mass, radius, pressure, luminosity, 
        #temperature, and density. Only the element corresponding to the differential equation (derivatives[n])
        #input into RK4 has the correct updated value.
        state[MASS_UNIT_INDEX,i] = state[MASS_UNIT_INDEX,i-1] + step_size
        state[RADIUS_UNIT_INDEX,i] = RK4(derivatives[0], state[:,i-1], step_size,extra_const_parameters)[RADIUS_UNIT_INDEX]
        state[PRESSURE_UNIT_INDEX,i] = RK4(derivatives[1], state[:,i-1], step_size,extra_const_parameters)[PRESSURE_UNIT_INDEX]
        state[LUMINOSITY_UNIT_INDEX,i] = RK4(derivatives[2], state[:,i-1], step_size,extra_const_parameters)[LUMINOSITY_UNIT_INDEX]
        state[TEMP_UNIT_INDEX,i] = RK4(derivatives[3], state[:,i-1], step_size,extra_const_parameters)[TEMP_UNIT_INDEX]
        state[DENSITY_UNIT_INDEX,i] = equation_of_state(state[PRESSURE_UNIT_INDEX,i], state[TEMP_UNIT_INDEX,i], extra_const_parameters)

    return state

if __name__ == "__main__":
    pass

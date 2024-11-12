import numpy as np
import scipy as sp;

def RK4(f,current, step_size):
    """
    Straightforward implementation of RK4 algorithm
    Inputs:
        f: derivative of dependent variable. Takes form f(y_n),
            where y_n encodes the current state of the system
            f should output a numpy array with size of y_n
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


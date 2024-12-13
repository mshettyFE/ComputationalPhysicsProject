import numpy as np
from Derivatives import gen_Jacobian
from DifferenceEqs import calc_g
from numpy.linalg import solve
from StateVector import StateVector

def NewtonRaphson(parameters, n_shells,max_iters, output_fname):
    state = StateVector(n_shells)
    for i in range(max_iters):
        print(i)
        Jac = gen_Jacobian(state,parameters)
        residual = calc_g(state, parameters)
        delta = solve(Jac,-residual)
        unstitched_delta = state.unstitch_vector(delta)
        print(unstitched_delta.max())
        
        if(state.update_state(unstitched_delta)):
            print("Update caused values to go negative")
            break
    state.save_state(output_fname)
    return state.state_vec
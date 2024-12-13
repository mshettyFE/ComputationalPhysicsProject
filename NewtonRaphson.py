import numpy as np
from Derivatives import gen_Jacobian
from DifferenceEqs import calc_g
from numpy.linalg import solve
from StateVector import StateVector

def NewtonRaphson(parameters, n_shells,max_iters):
    state = StateVector(n_shells, test_data=True) # For now, use test data
    for i in range(max_iters):
        Jac = gen_Jacobian(state,parameters)
        residual = calc_g(state, parameters)
        delta = solve(Jac,-residual) # Don't forget the negative sign!
        unstitched_delta = state.unstitch_vector(delta)
        if(state.update_state(unstitched_delta)):
            print("Update caused values to go negative")
            break
    return state.state_vec
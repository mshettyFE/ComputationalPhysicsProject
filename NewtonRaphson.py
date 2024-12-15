import numpy as np
from Derivatives import gen_Jacobian
from DifferenceEqs import calc_g
from numpy.linalg import solve
from StateVector import StateVector

def NewtonRaphson(parameters, n_shells,max_iters, output_fname="REPLACEME.txt", verbose=False):
    state = StateVector(n_shells)
    for i in range(max_iters):
        print(i)
        Jac = gen_Jacobian(state,parameters, verbose)
        inv_Jac = np.linalg.inv(Jac)
        residual = calc_g(state, parameters)
        delta = -np.matmul(inv_Jac,residual)
        unstitched_delta = state.unstitch_vector(delta)
        if(verbose):
            print("inv_Jac_min", inv_Jac.min())
            print("inv_Jac_max", inv_Jac.max())
            print("d_min",unstitched_delta.min())
            print("d_max",unstitched_delta.max())
            print("res_min",(-residual).min())
            print("res_max",(-residual).max())
        
        if(state.update_state(unstitched_delta)):
            print("Update caused values to go negative")
            break
    state.save_state(output_fname)
    return state.state_vec
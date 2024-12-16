import jax.numpy as jnp
from jax import jacfwd
from DifferenceEqs import calc_jax_g
from StateVector import StateVector

def is_invertible(a):
    return a.shape[0] == a.shape[1] and jnp.linalg.matrix_rank(a) == a.shape[0]


def NewtonRaphson(parameters, n_shells,max_iters, output_fname="REPLACEME.npy", verbose=False):
    state = StateVector(n_shells)
    for i in range(max_iters):
        print(i)
        Jacobian = jacfwd(calc_jax_g)(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        Jac = state.stich_jacobian(Jacobian)
        residual = calc_jax_g(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        resi = state.stitch_vector(residual)
        inv_Jac = jnp.linalg.inv(Jac)
        delta = -jnp.matmul(inv_Jac,resi)
        if(verbose):
            print("inv_Jac_min", inv_Jac.min())
            print("inv_Jac_max", inv_Jac.max())
            print("d_min",delta.min())
            print("d_max",delta.max())
            print("res_min",(-resi).min())
            print("res_max",(-resi).max())
        unstitched_delta = state.unstitch_vector(delta)
        if(state.update_state(unstitched_delta)):
            print("Update caused values to go negative")
            break
    state.save_state(output_fname)
    return state.state_vec
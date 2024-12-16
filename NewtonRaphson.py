import jax.numpy as jnp
from jax import jacfwd
from DifferenceEqs import calc_jax_g
from StateVector import StateVector, DataGenMode

def NewtonRaphsonWrapper(params):
    attempt_counter = 1
    while(True):
        out, errors, initial = NewtonRaphson(params, 100,25, DataGenMode.RANDOM)
        n_iters = len(errors)
        final_err = errors[-1]
        output_fname="REPLACEME.npy"
        if (final_err< 1):
            print("Attempt: ", attempt_counter,"Error", final_err, "Niters", n_iters)
            print("Converged with Error: ", final_err)
            print(errors)
            if(out.update_state(jnp.zeros(out.state_vec.shape[0]))):
                print("Some values were negative: ", errors)
                continue
            out.save_state(output_fname)
            jnp.save("InitialConds.npy",initial)
            break
        print("Attempt: ", attempt_counter, "Error",final_err,"Niters", n_iters)
        attempt_counter += 1

def NewtonRaphson(parameters, n_shells,max_iters, gen_mode, verbose=False):
    state = StateVector(n_shells, gen_mode)
    initial = state.state_vec
    errs = []
    prev_err = jnp.inf
    for i in range(max_iters):
        n_iters = i
        Jacobian = jacfwd(calc_jax_g)(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        Jac = state.stich_jacobian(Jacobian)
        residual = calc_jax_g(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        resi = state.stitch_vector(residual)
        delta = jnp.linalg.solve(Jac,-resi)
        err = jnp.sum(jnp.abs(resi))
        errs.append(err)
        if(verbose):
            print("d_min",delta.min())
            print("d_max",delta.max())
            print("res_min",(-resi).min())
            print("res_max",(-resi).max())
            print("total err",err)
        unstitched_delta = state.unstitch_vector(delta)
        if(state.update_state(unstitched_delta)):
            print("Update caused values to go negative")
            break
        if jnp.isnan(err):
            break
        elif(jnp.abs(err-prev_err)< 1E-4):
            break
        prev_err = err
    return state, errs, initial
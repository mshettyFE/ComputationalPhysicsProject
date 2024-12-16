import jax.numpy as jnp
from jax import jacfwd
from DifferenceEqs import calc_jax_g
from StateVector import StateVector, DataGenMode

def NewtonRaphsonWrapper(params,n_shells,n_steps, err_bound):
    attempt_counter = 1
    while(True):
        out, errors, initial = NewtonRaphson(params, n_shells,n_steps, DataGenMode.RANDOM, verbose=False)
        n_iters = len(errors)
        final_err = errors[-1]
        output_fname="REPLACEME.npy"
        if (final_err< err_bound):
            print("Attempt: ", attempt_counter,"Error", final_err, "Niters", n_iters)
            print("Converged with Error: ", final_err)
            print(errors)
            if(out.update_state(jnp.zeros(out.state_vec.shape[0]))==0):
                print("Some values were negative: ", errors)
                continue
            out.save_state(output_fname)
            jnp.save("InitialConds.npy",initial)
            break
        print("Attempt: ", attempt_counter, "Error",final_err,"Niters", n_iters)
        attempt_counter += 1
    return out, errors, initial

def NewtonRaphson(parameters, n_shells,max_iters, gen_mode, verbose=False):
    state = StateVector(n_shells, gen_mode)
    initial = state.state_vec
    errs = []
    prev_err = jnp.inf
    for i in range(max_iters):
        Jacobian = jacfwd(calc_jax_g)(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        Jac = state.stich_jacobian(Jacobian)
        residual = calc_jax_g(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        resi = state.stitch_vector(residual)
        inv_Jac = jnp.linalg.inv(Jac)
#        eigenval, eigenvecs = jnp.linalg.eig(Jac)
#        eigenval  = jnp.abs(eigenval)
#        condition = eigenval.max()/eigenval.min()
        delta = jnp.matmul(inv_Jac, -resi)
        err = jnp.sum(jnp.abs(resi))
        errs.append(err)
        negative_flag = False
        unstitched_delta = state.unstitch_vector(delta)
        for i in range(0,10):
            negative_flag = False
            scaling = 1/jnp.power(10,i)
            scaled_delta = unstitched_delta*scaling
            if(state.update_state(scaled_delta)):
                break
            else:
                negative_flag = True
        if(verbose):
#            print("Iter", i,"Scaling", scaling,"Error",err,"Condition", condition)
            print("Iter", i,"Scaling", scaling,"Error",err)
        if(negative_flag):
            if(verbose):
                print("Step causes update to be negative")
            break
        if jnp.isnan(err):
            break
        elif(jnp.abs(err-prev_err)< 1E-4):
            break
        prev_err = err
    return state, errs, initial
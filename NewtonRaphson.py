import jax.numpy as jnp
from jax import jacfwd
from DifferenceEqs import calc_jax_g, calc_jax_g_log
from StateVector import StateVector, DataGenMode

def NewtonRaphsonWrapper(params,n_shells,n_steps, err_bound, output_fname="REPLACEME.npy", verbose = True):
    """
        Wrapper around random initial condition for NewtonRaphson
        Inputs:
            params: Dictionary of external parameters
            n_shells: number of shells which need to be generated
            err_bound: the error threshold at which to terminate
            output_fname = what fine to write the state vector to
        Output:
            out: final state vector
            errors: list of all the errors encountered during the process
            initial: the initial conditions which generated the final output
    """
    attempt_counter = 1
    while(True):
        out, errors, initial = NewtonRaphson(params, n_shells,n_steps, DataGenMode.RANDOM, verbose=False)
        n_iters = len(errors)
        final_err = errors[-1]
        if (final_err< err_bound):
            if(verbose):
                print("Attempt: ", attempt_counter,"Error", final_err, "Niters", n_iters)
                print("Converged with Error: ", final_err)
            if(out.update_state(jnp.zeros(out.state_vec.shape[0]))==0):
                print("Some values were negative: ", errors)
                continue
            out.save_state(output_fname)
            jnp.save("InitialConds.npy",initial)
            break
        if(verbose):
            print("Attempt: ", attempt_counter, "Error",final_err,"Niters", n_iters)
        attempt_counter += 1
    return out, errors, initial

def NewtonRaphson(parameters, n_shells,max_iters, gen_mode, verbose=False):
    """
        Implements multidimensional NR method to iteratively find solution
        Inputs:
            parameters: Dictionary of external parameters
            n_shells: number of shells which need to be generated
            max_iters: Max number of iterations to take
            gen_mode = How to instantiate the state vector
        Output:
            out: final state vector
            errors: list of all the errors encountered during the process
            initial: the initial conditions which generated the final output        
    """
    state = StateVector(n_shells, gen_mode)
    initial = state.state_vec
    errs = []
    prev_err = jnp.inf
    for i in range(max_iters):
        # Calculate Jacobian and g
        if(state.data_gen_mode==DataGenMode.LOG):
            Jacobian = jacfwd(calc_jax_g_log)(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
            residual = calc_jax_g_log(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        else:
            Jacobian = jacfwd(calc_jax_g)(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
            residual = calc_jax_g(state.state_vec, state.starting_indices, state.n_shells, state.dm, parameters)
        # Calculate updates and errors
        Jac = state.stich_jacobian(Jacobian)
        resi = state.stitch_vector(residual)
        inv_Jac = jnp.linalg.inv(Jac)
        delta = jnp.matmul(inv_Jac, -resi)
        err = jnp.sum(jnp.abs(resi))
        errs.append(err)
        unstitched_delta = state.unstitch_vector(delta)
        # Try to apply update such that state_vec remains valid
        if (state.data_gen_mode==DataGenMode.LOG):
            scaling = 1/jnp.power(10,5)
            scaled_delta = unstitched_delta*scaling
            state.update_state(scaled_delta, check_neg=False)
        else:
            for i in range(0,15):
                scaling = 1/jnp.power(2,i)
                scaled_delta = unstitched_delta*scaling
                if (state.update_state(scaled_delta)):
                    break
        # Debugging code
        if(verbose):
            eigenval, eigenvecs = jnp.linalg.eig(Jac)
            eigenval  = jnp.abs(eigenval)
            condition = eigenval.max()/eigenval.min()
            print("Iter", i,"Scaling", scaling,"Error",err,"Condition", condition)
#            print("Iter", i,"Error",err)
        # Sanity check
        if jnp.isnan(err):
            break
        # termination condition
        elif(jnp.abs(err-prev_err)< 1E-4):
            break
        prev_err = err
    return state, errs, initial
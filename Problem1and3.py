import NewtonRaphson
import Utilities
import jax.numpy as jnp
from StateVector import DataGenMode

if __name__ == "__main__":
    mass = Utilities.M_sun*1
    rad = Utilities.R_sun
    params = Utilities.generate_extra_parameters(mass, rad, Utilities.E_0_sun, Utilities.kappa_0_sun, Utilities.mu_sun)
    print(params)
    scales = Utilities.UnitScalingFactors(mass, rad)
    print(scales)
#    state, err, initial = NewtonRaphson.NewtonRaphsonWrapper(params,100, 100, 1E4)
    state, err, initial = NewtonRaphson.NewtonRaphson(params,100, 100, DataGenMode.LOG, verbose=True)
    state.save_state("REPLACEME.npy")
    jnp.save("InitialConds.npy",initial)

import NewtonRaphson
import Utilities
import jax.numpy as jnp

if __name__ == "__main__":
    params = Utilities.generate_extra_parameters(Utilities.M_sun, Utilities.R_sun, Utilities.E_0_sun, Utilities.kappa_0_sun, Utilities.mu_sun)
    NewtonRaphson.NewtonRaphson(params, 100,10)
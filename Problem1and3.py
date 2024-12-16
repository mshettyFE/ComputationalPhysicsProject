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
    NewtonRaphson.NewtonRaphsonWrapper(params)
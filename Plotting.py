import jax.numpy as jnp
from StateVector import StateVector, StateVectorVar
import matplotlib.pyplot as plt

if __name__ == "__main__":
    s =  StateVector(50, True)
    s.load_state("REPLACEME.npy")
    rad = s.extract_variable(StateVectorVar.RADIUS)
    pres = s.extract_variable(StateVectorVar.PRESSURE)
    print(s.state_vec.shape)
    r_start = s.starting_indices[StateVectorVar.RADIUS]
    p_start  =s.starting_indices[StateVectorVar.LUMINOSITY]
    plt.scatter(s.state_vec[r_start:r_start+s.n_shells], s.state_vec[p_start:p_start+s.n_shells])
    plt.show()
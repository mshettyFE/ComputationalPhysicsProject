import jax.numpy as jnp
from StateVector import StateVector, StateVectorVar

class TestStateVec:
    def test_stitch(self):
        sv = StateVector(4, True)
        # Overwrite internal state with more tractable state
        sv.state_vec = jnp.arange(4*4)
        sv.starting_indices = sv.gen_starting_index()
        output = sv.stitch_vector(sv.state_vec)
        expected = jnp.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15], dtype=jnp.float64)
        assert (output==expected).all()
        # Make sure that it scales
        sv = StateVector(10, True)
        # Overwrite internal state with more tractable state
        sv.state_vec = jnp.arange(4*10)
        sv.starting_indices = sv.gen_starting_index()
        output = sv.stitch_vector(sv.state_vec)
        expected = jnp.array( [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39], dtype=jnp.float64)
        assert (output==expected).all()
    
    def test_unstitch(self):
        sv = StateVector(4, True)
        # Overwrite internal state with more tractable state
        sv.state_vec = jnp.arange(4*4)
        sv.starting_indices = sv.gen_starting_index()
        rsv = jnp.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15], dtype=jnp.float64)
        output = sv.unstitch_vector(rsv)
        expected = jnp.array([0, 1,  2,  3,  4,  5,  6, 0, 8,  9, 10,0,0 ,13, 14, 15], dtype=jnp.float64)
        assert (output==expected).all()
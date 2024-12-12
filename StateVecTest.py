import numpy as np
from StateVector import StateVector, StateVectorVar

class TestStateVec:
    def test_stitch(self):
        sv = StateVector(4, True)
        # Overwrite internal state with more tractable state
        sv.state_vec = np.arange(4*4)
        sv.starting_indices = sv.gen_starting_index()
        output = sv.stitch_vector()
        expected = np.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15], dtype=np.float64)
        assert (output==expected).all()
        # Make sure that it scales
        sv = StateVector(10, True)
        # Overwrite internal state with more tractable state
        sv.state_vec = np.arange(4*10)
        sv.starting_indices = sv.gen_starting_index()
        output = sv.stitch_vector()
        expected = np.array( [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39], dtype=np.float64)
        assert (output==expected).all()
    
    def test_unstitch(self):
        sv = StateVector(4, True)
        # Overwrite internal state with more tractable state
        sv.state_vec = np.arange(4*4)
        sv.starting_indices = sv.gen_starting_index()
        rsv = np.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15], dtype=np.float64)
        output = sv.unstitch_vector(rsv)
        expected = np.array([0, 1,  2,  3,  4,  5,  6, 0, 8,  9, 10,0,0 ,13, 14, 15], dtype=np.float64)
        assert (output==expected).all()

    def test_interp(self):
        sv = StateVector(4, True)
        output = sv.interpolate(StateVectorVar.RADIUS)
        expected = np.array([1,3,5])
        assert (output==expected).all()

        output = sv.interpolate(StateVectorVar.PRESSURE)
        expected = np.array([1.5, 4.5, 7.5])
        assert (output==expected).all()

        output = sv.interpolate(StateVectorVar.TEMP)
        expected = np.array([2.5,  7.5, 12.5])
        assert (output==expected).all()

        output = sv.interpolate(StateVectorVar.LUMINOSITY)
        expected = np.array([3.5, 10.5, 17.5])
        assert (output==expected).all()
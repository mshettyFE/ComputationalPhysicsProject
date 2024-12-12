import numpy as np
import Utilities

class TestUtilities:
    def test_stitch(self):
        sv = np.arange(4*4)
        starting = Utilities.gen_starting_index(sv)
        output = Utilities.stitch_vector(sv, starting)
        expected = np.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15], dtype=np.float64)
        assert (output==expected).all()
        # Make sure that it scales
        sv = np.arange(4*10)
        starting  = Utilities.gen_starting_index(sv)
        output = Utilities.stitch_vector(sv, starting)
        expected = np.array( [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39], dtype=np.float64)
        assert (output==expected).all()
    
    def test_unstitch(self):
        original  = np.arange(4*4)
        rsv = np.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15], dtype=np.float64)
        starting = Utilities.gen_starting_index(original)
        output = Utilities.unstitch_vector(rsv, starting)
        expected = np.array([0, 1,  2,  3,  4,  5,  6, 0, 8,  9, 10,0,0 ,13, 14, 15], dtype=np.float64)
        assert (output==expected).all()

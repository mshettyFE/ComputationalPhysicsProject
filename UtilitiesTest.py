import numpy as np
import Utilities

class TestUtilities:
    def test_stich(self):
        sv = np.arange(4*4)
        starting = Utilities.gen_starting_index(sv)
        output = Utilities.stich_vector(sv, starting)
        expected = np.array([ 1,  2,  3,  4,  5,  6,  8,  9, 10, 13, 14, 15])
        assert (output==expected).all()
        # Make sure that it scales
        sv = np.arange(4*10)
        starting  = Utilities.gen_starting_index(sv)
        output = Utilities.stich_vector(sv, starting)
        expected = np.array( [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39])
        assert (output==expected).all()
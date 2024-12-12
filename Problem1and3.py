import NewtonRaphson
import Utilities
import numpy as np

if __name__ == "__main__":
    k = 4
    sv = np.arange(4*10)
    starting = Utilities.gen_starting_index(sv)
    print(sv)
    print(Utilities.stich_vector(sv, starting))
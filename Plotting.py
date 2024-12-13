import numpy as np
from StateVector import StateVector

if __name__ == "__main__":
    s =  StateVector(50, True)
    s.load_state("REPLACEME.txt")
    print(s.state_vec.shape)
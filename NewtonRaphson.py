import numpy as np
from Derivatives import gen_Jacobian
from numpy.linalg import solve
import Utilities

def NewtonRaphson(n_shells,max_iters):
    cur_state, state_indicies = Utilities.gen_state_vector(n_shells)
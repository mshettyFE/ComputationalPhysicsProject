import NewtonRaphson
import Utilities
import numpy as np

if __name__ == "__main__":
    params = {
        "mu": 1,
        "E0_prime": 1,
        "k0_prime": 1
    }
    NewtonRaphson.NewtonRaphson(params, 500,10)
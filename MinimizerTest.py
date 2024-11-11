import numpy as np
import pytest
import Minimizer

def derivative_func(state):
    return -state


class TestMinimizer:
    def test_RK4(self):
        """
            Use the IVP of dy/dt = -y. y(0) = 1.
            This should have the solution y(t) = e(-t)
            We make 1000 steps, and then check against the analytic result
        """
        # Extraneous 0 after 1 because actual problem has 5 dependent variables
        state = np.array([0,1 ])
        iterations = 1000
        start = 0
        stop  = 10
        step_size = (stop-start)/iterations
        tolerance = 1E-1;
        x = np.arange(0,10, step_size)
        assert(x.shape[0] == iterations)
        for i in range(iterations):
            actual_val = np.exp(-x[i])
            error = np.abs(actual_val-state[1])
            assert (error < tolerance)
            state = Minimizer.RK4(derivative_func, state, step_size)


if __name__ == "__main__":
    pass


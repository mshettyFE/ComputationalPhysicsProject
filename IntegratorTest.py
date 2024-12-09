
import numpy as np
import pytest
import Integrator

def derivative_func(state, extra):
    return -state


class TestMinimizer:
    def test_RK4(self):
        """
            Use the IVP of dy/dt = -y. y(0) = 1.
            This should have the solution y(t) = e(-t)
            We make 1000 steps, and then check against the analytic result at each step
        """
        # Extraneous 0 after 1 because actual problem has 5 dependent variables
        state = np.array([0,1,0,0,0,0 ])
        iterations = 1000
        start = 0
        stop  = 10
        step_size = (stop-start)/iterations
        tolerance = 1E-9
        x = np.arange(0,10, step_size)
        assert(x.shape[0] == iterations)
        for i in range(iterations):
            actual_val = np.exp(-x[i])
            error = np.abs(actual_val-state[1])
            assert (error < tolerance)
            state = state+Integrator.RK4(derivative_func, state, step_size,{})
    def test_convergence(self):
        """
            Use the IVP of dy/dt = -y. y(0) = 1.
            This should have the solution y(t) = e(-t)
            We make 1000 steps, then 1000 steps and then check against the analytic result at the end
            We expect the ratio of the final values of the different step sizes to be ~1E4, since RK4 is 4th order
        """
        start = 0
        stop  = 10

        state = np.array([0,1,0,0,0,0 ])
        iterations = 100
        step_size = (stop-start)/iterations
        for i in range(iterations):
            state = state+  Integrator.RK4(derivative_func, state, step_size,{})
        one_tenth = state[1]

        state = np.array([0,1,0,0,0,0 ])
        iterations = 1000
        step_size = (stop-start)/iterations
        for i in range(iterations):
            state = state+  Integrator.RK4(derivative_func, state, step_size,{})
        one_hundred = state[1]

        expected_value = np.exp(-stop)
        err_ten = np.abs(expected_value-one_tenth)
        err_hund = np.abs(expected_value-one_hundred)
        print(err_ten, err_hund,expected_value)
        assert(np.floor(np.log10(err_ten/err_hund)) == 4)


if __name__ == "__main__":
    pass


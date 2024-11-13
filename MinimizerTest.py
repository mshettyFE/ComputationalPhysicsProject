
import numpy as np
import pytest
import Minimizer

def test_constraints():
    cnst = Minimizer.equation_of_state_constraint(-1000)
    expected_A = np.array(
        [[1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,1000],
        [0,0,0,1,0],
        [0,0,0,0,1]]
    )
    assert(np.array_equal(cnst.A, expected_A))

if __name__ == "__main__":
    pass
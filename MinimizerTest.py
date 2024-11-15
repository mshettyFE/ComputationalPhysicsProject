
import numpy as np
import pytest
import Minimizer

class TestMinimizer:
    def test_minimizer(self):
    # Make sure that minimmizer can be run from end to end
        ns = 1000
        out  = Minimizer.run_minimizer(10,10,ns,1,2,3,4,5)

if __name__ == "__main__":
    pass
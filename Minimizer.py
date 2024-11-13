import numpy as np
import scipy as sp
import Integrator

def equation_of_state_constraint(alpha):
# x = [m',r',P',L',T'] with shape of 5x1
# The nominal bounds on the variables are positive numbers
    lb = np.zeros((5,))
    ub = np.zeros((5,))
    ub.fill(np.inf)
# Only real constraint is p'-\gamma T' = 0
# This takes care of RHS of this constraint
    lb[2] = 0
    ub[2] = 0
    A = np.zeros((5,5))
    A[0][0] = 1 # m' term
    A[1][1] = 1 # r' term
# LHS of equation of state
    A[2][2] = 1
    A[2][4] = -alpha
    A[3][3] = 1 # L' term
    A[4][4] = 1 # T' term
    return sp.optimize.LinearConstraint(A, lb,ub, True) # want constraints to hold for all mass steps

def loss_function(estimator_guess, *args):
# We assume that estimator has dimensions 2x1
# Temperature and pressure normally can't be negative
    assert(estimator_guess[0] >= 0) # temperature
    assert(estimator_guess[1] >= 0) # pressure
# Want temperature and pressure to be 0 at boundaries. These variables are mostly just for clarity
    expected_pressure = np.array((0))
    expected_temp = np.array((0))
# *args should of the form  [ODE solver, num_iters]
    assert(len(args) == 2)
# encode the boundary conditions of m'= L' = r'=0, plug put in the initial temp and pressure guesses
    initial_conds = np.array(  (0,0,estimator_guess[0], 0, estimator_guess[1]) )
# use the ODE solver to propagate the initial conditions to the final state

# This signature might be wrong though since it hasn't been developed yet
    final_conditions = args[0](initial_conds, args[1])

# final_conditions = [m,r,P,L,T]
    return   np.pow((final_conditions[2]-expected_temp),2) + np.pow(final_conditions[4]-expected_pressure,2)

def run_minimizer():
    Initial_T = 10
    Initial_P = 10
    gamma = 10
    x0 = np.array([Initial_T, Initial_P])
    solver = None
#    solver = Integrator.ODE
    cnstrs = equation_of_state_constraint(gamma)
    return sp.optimize.minimize(loss_function,x0, args=(solver), constraints= cnstrs, method="COBYLA")

if __name__ == "__main__":
    pass
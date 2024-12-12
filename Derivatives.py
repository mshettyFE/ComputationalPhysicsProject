import numpy as np

# -------------------------------------------------------------------
# NOTE: Don't forget that density(\rho) is a function of Pressure(P) and Temperature(T)
# Therefore, you need to remember to take those derivatives!
# Remember: \rho_{half} = Utilities.equation_of_state(P_{half},T_{half}) = \mu (P_{k+1}-P_{k})/(T_{k+1}-T_{k})
# -------------------------------------------------------------------

# NOTE: dm = m_{k+1}-m_{k}. Assume this is a constant (ie. shells sizes are the same throughout)

# NOTE: This file houses the sub-matrix generator for the Jacobian, and the Jacobian calculator
# There are 16 functions, with each one dedicated to a particular sub block

# -------------------------------------------------------------------
# NOTE: We expect all submatrix calculations to have the same signature:
# state_vector: 4k dimensional vector representing the current state of the system
# starting_indices: Dictionary containing the offset where each variable gets started at
    # This is 0 for radius, k for pressure, 2k for temp, and 3k for lum.
    # This is a dictionary so that there is only one place to look of things aren't offset properly
# n_shells: number of shells. Just so that we don't need to parse n_shells from state_vector for all functions

# Output: We expect the output of each function to be a nxn matrix
# where n = (k-1), where k is the number of shells
# Stitching everything together should give a 4(k-1) square matrix
# To take into account the boundary conditions, we use stitch_vector on the state vector
# -------------------------------------------------------------------


# Difference equation for radius
# (r_{k+1}-r_{k})/(dm)- 1/(4*pi*r^{2}_{half}* \rho_{half})

# Radius
def Jac_block_00(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Pressure
def Jac_block_01(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Temperature
def Jac_block_02(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Luminosity
def Jac_block_03(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Difference equation for Pressure
# (P_{k+1}-P_{k})/(dm) +  (dm/2)/(4*pi*r^{4}_{half})

# Radius
def Jac_block_10(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Pressure
def Jac_block_11(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Temperature
def Jac_block_12(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Luminosity
def Jac_block_13(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Difference equation for Temperature
# (T_{k+1}-T_{k})/(dm)+ \kappa_0 \rho_{half}* L_{half}/r^{4}_{half}/T^{6.5}_{half}

# Radius
def Jac_block_20(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Pressure
def Jac_block_21(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Temperature
def Jac_block_22(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Luminosity
def Jac_block_23(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Difference equation for luminosity
# (L_{k+1}-L_{k})/(dm)- \epsilon_0 \rho_{half}*T_{half}^{4}=g

# Radius
def Jac_block_30(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Pressure
def Jac_block_31(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Temperature
def Jac_block_32(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

# Luminosity
def Jac_block_33(state_vector,starting_indicies, n_shells):
    output  = output  = np.zeros((n_shells-1, n_shells-1))
    return output

def gen_Jacobian(state_vector, starting_indices):
    """
        Outputs the Jacobian of the residual function needed for Newton-Raphson
        Input:
            state_vector: 4k dimensional vector of the current variable states
            Assumes that state_vector has the following form:
            <r_0, r_1,...r_{k-1}, P_0,P_1,...P_{k-1}, T_0,...T_{k-1}, L_0,...L_{k-1}>, where k is the number of shells
        Output:
            nxn Jacobian matrix, where n = 4*(k-1), where k is the number of shells
            Need to use Utilities.stich_vector to make this Jacobian useful in NR method
    """
    dim = state_vector.shape[0]
    assert(dim != 0)
    assert(dim%4==0)
    n_shells = int(dim/4)
    output_dim = 4*(n_shells-1)
    Jac = np.zeros((output_dim, output_dim))
    J00 = Jac_block_00(state_vector, starting_indices,n_shells)
    J01 = Jac_block_01(state_vector, starting_indices,n_shells)
    J02 = Jac_block_02(state_vector, starting_indices,n_shells)
    J03 = Jac_block_03(state_vector, starting_indices,n_shells)
    J0 = np.concatenate([J00, J01, J02, J03], axis=1)

    J10 = Jac_block_10(state_vector, starting_indices,n_shells)
    J11 = Jac_block_11(state_vector, starting_indices,n_shells)
    J12 = Jac_block_12(state_vector, starting_indices,n_shells)
    J13 = Jac_block_13(state_vector, starting_indices,n_shells)
    J1 = np.concatenate([J10, J11, J12, J13], axis=1)

    J20 = Jac_block_20(state_vector, starting_indices,n_shells)
    J21 = Jac_block_21(state_vector, starting_indices,n_shells)
    J22 = Jac_block_22(state_vector, starting_indices,n_shells)
    J23 = Jac_block_23(state_vector, starting_indices,n_shells)
    J2 = np.concatenate([J20, J21, J22, J23], axis=1)

    J30 = Jac_block_30(state_vector, starting_indices,n_shells)
    J31 = Jac_block_31(state_vector, starting_indices,n_shells)
    J32 = Jac_block_32(state_vector, starting_indices,n_shells)
    J33 = Jac_block_33(state_vector, starting_indices,n_shells)
    J3 = np.concatenate([J30, J31, J32, J33], axis=1)
    
    out = np.concatenate([J0,J1,J2,J3], axis=0)
    return out


if __name__=="__main__":
    pass
import numpy as np
import sys
from enum import Enum

# values of fundamental constants in SI units
G = np.float64(6.6743E-11) # (Nm^{2})/(kg^{2})
StefanBoltz = np.float64(5.67E-8) # W/(m^{2}K^{4})
Boltzman = np.float64(1.380649E-23) # J/K
m_p = np.float64(1.67262192E-27) # kg

# Parameters for the Sun
M_sun = 1.989E30 # kg
R_sun  = 6.9634E8 # m
mu_sun = np.float64(0.6)
E_0_sun = 1.8E-26 # m^5/(kg s^3*K^4)
kappa_0_sun = 3E-2 # m^2/kg

class ScaleIndex(Enum):
    RADIUS = 0
    PRESSURE = 1
    TEMP = 2
    LUMINOSITY = 3
    MASS = 4
    DENSITY= 5
    TIME = 6

class StateVectorVar(Enum):
    RADIUS=0
    PRESSURE=1
    TEMP=2
    LUMINOSITY=3

def UnitScalingFactors( M_0, R_0):
    """
    Returns the scaling factors to convert the unitless 
    numbers in the sim to physical units.
    Inputs:
        M_0: Mass Scale
        R_0: Length Scale
    Output:
        Dictionary of {ScaleIndex: np.float64}, where ScaleIndex denotes which unit, and np.float64 denotes scale factor
        Use the *_UNIT_INDEX variables to get the corresponding scalings
    """
    assert(R_0 > 0)
    assert(M_0 > 0)
    #The "out" variables are the coefficients which multiply the scaled variables and generate the original unit variables.
    R_out = np.float64(R_0)
    M_out = np.float64(M_0)
    rho_out = M_0/(np.power(R_0,3))
    t_0 = np.sqrt(np.power(R_0,3) / (G*M_0))
    P_out = M_0/(R_0*np.power(t_0,2))
    L_out = (M_0*np.power(R_0,2)) / (np.power(t_0,3))
    T_out = (m_p*np.power(R_0,2)) / (np.power(t_0,2)*Boltzman) # Using m_p here since T_out is the temp scale on a particle level, not on a star level
    out = {}
    out[ScaleIndex.MASS] = M_out
    out[ScaleIndex.RADIUS] = R_out
    out[ScaleIndex.DENSITY] = rho_out
    out[ScaleIndex.PRESSURE] = P_out
    out[ScaleIndex.LUMINOSITY] = L_out
    out[ScaleIndex.TEMP] = T_out
    out[ScaleIndex.TIME] = t_0
    return out

def generate_extra_parameters(M_0, R_0, E_0, kappa, mu):
    """
        Given the unitful parameters of the problem, generate the unitless constants to be used in the simulation
    Inputs:
        R_0: Length Scale (maximum radius)
        M_0: Mass Scale (total mass)
        epsilon_0: nuclear energy generation constant for luminosity equation (m^5/(kg s^3*K^4)) dependent on main fusion reaction (proton-proton in the Sun)
        kappa: opacity parameter for temperature equation (m^2/kg)
        mu: mean molecular weight in units of proton mass
    Output:
        extra_const_params: python dictionary containing the converted constant parameters
    """
    scale_factors = UnitScalingFactors(M_0, R_0)
    T0 = scale_factors[ScaleIndex.TEMP]
    t0 = scale_factors[ScaleIndex.TIME]
    L0 = scale_factors[ScaleIndex.LUMINOSITY]

    ep_prefactor = np.power(R_0,5)*np.power(M_0,-1)*np.power(T0,-4)*np.power(t0,-3)
    new_ep = E_0/ep_prefactor


    kp_prefactor = np.power(M_0,-2)*np.power(R_0,5)*np.power(T0,3.5)
    scaled_stefan = StefanBoltz*(np.power(R_0,2),np.power(T0,4))/L0
    kp_const_prefactor = (3/(16*scaled_stefan*np.power(4*np.pi,2) ))
    new_kp = kp_const_prefactor*kappa/kp_prefactor

    
    extra_const_params = {
        "mu": mu,
    "E0_prime": new_ep,
    "k0_prime": new_kp,
    }

    return extra_const_params

def equation_of_state(p_state,t_state, extra_const_params):
    """
        generate the density at the shell boundaries given the pressure and temperature
        Input:
            p_state: k-1 np array of dimensionless pressures where k is number of shells
            t_state: np array of dimensionless temp. Same size as p_state
            extra_const_params: constant parameters dictionary
        Output:
            array of densities. same size as p_state
    """
    return (p_state*extra_const_params["mu"])/t_state

def nuclear_energy(rho_prime, T_prime, extra_const_params):
    """
        generate the nuclear energy production rate given the density and temperature
        Input:
            rho_prime: array of size k-1 of dimensionless density
            T_prime: array of dimensionless temperature. Same size as rho_prime
        Output:
            E_prime: dimensionless energy rate
    """
    E_prime = (extra_const_params["E0_prime"])*rho_prime*np.power(T_prime,4)
    return E_prime

# State vector manipulation

def gen_state_vector(n_shells, test=False):
    """
        Input: 
        Output:
            state_vector: 4k dimensional vector
            Assume state_vector has the following form:
                <r_0, r_1,...r_{k-1}, P_0,P_1,...P_{k-1}, T_0,...T_{k-1}, L_0,...L_{k-1}>
            starting_indices:
                Dictionary of starting indices for each variable in state vector
    """
    if(test):
        rad = 2*np.ones(n_shells)
        pres = 3*np.ones(n_shells)
        temp = 5*np.ones(n_shells)
        lum = 7*np.ones(n_shells)
        output = np.concatenate([rad, pres, temp, lum], axis=None)
    else:
        #TODO: Makes these guesses realistic. For now, just panic
        print("Make better guesses please")
        sys.exit(1)
    starting_indices = gen_starting_index(output)
    return output, starting_indices

def average_adjacent(vector):
    """
        generate average of adjacent elements in a vector
        Input:
            vector: n dimensional np array
        Output:
            n-1 dimensional np array    
    """
    first_couple = vector[:-1] # Exclude last
    last_couple = vector[1:] # Exclude first
    return (first_couple+last_couple)/2

def gen_starting_index(state_vector):
    """
        Assumes that state_vector was generated in gen_state_vector()
        Input:
            state_vector: 4k dimensional array where k is the number of mass shells
        Output:
            starting_indices: a map from StateVectorVar to the starting index in state_vector
    """
    dim = state_vector.shape[0]
    assert(dim != 0)
    assert(dim%4==0)
    n_shells = int(dim/4)
    starting_indices = {}
    starting_indices[StateVectorVar.RADIUS] = 0
    starting_indices[StateVectorVar.PRESSURE] = n_shells
    starting_indices[StateVectorVar.TEMP] = n_shells*2
    starting_indices[StateVectorVar.LUMINOSITY] = n_shells*3
    return starting_indices

def stitch_vector(state_vector, starting_indices):
    """
        Given a state vector, remove the elements which do not change (ie. the 0 boundary conditions).
        This involves removing r_0,P_{k-1}, T_{k-1},L_0 where k is the number of shells.
        This is done to preserve the boundary conditions of the system (the removed elements are all 0).
        By removing these elements, we prevent the Jacobian from going singular.
        Assumes that state_vector has the following form:
        <r_0, r_1,...r_{k-1}, P_0,P_1,...P_{k-1}, T_0,...T_{k-1}, L_0,...L_{k-1}>, where k is the number of shells
        Input:
            state_vector: 4k dimensional array where k is the number of mass shells
            starting_indices: The starting index of each variable (use gen_starting_index to produce)
        Output:
            out_vec: 4*(k-1) dimensional array
    """
    dim = state_vector.shape[0]
    assert(dim != 0)
    assert(dim%4==0)
    n_shells = int(dim/4)
    output = np.zeros(4*(n_shells-1))

    starting_rad = starting_indices[StateVectorVar.RADIUS]
    starting_p = starting_indices[StateVectorVar.PRESSURE]
    starting_temp = starting_indices[StateVectorVar.TEMP]
    starting_lum = starting_indices[StateVectorVar.LUMINOSITY]

    rad_part = state_vector[starting_rad+1:starting_rad+n_shells] # exclude r_0
    pressure_part = state_vector[starting_p:starting_p+n_shells-1] # exclude P_{k-1}
    temp_part = state_vector[starting_temp:starting_temp+n_shells-1] # exclude T_{k-1}
    lum_part = state_vector[starting_lum+1:starting_lum+n_shells] # exclude L_0

    # Need to subtract off from the starting index as you go up to have proper indexing
    output[starting_rad:starting_rad+n_shells-1] = rad_part
    output[starting_p-1:starting_p+n_shells-2] = pressure_part 
    output[starting_temp-2:starting_temp+n_shells-3] = temp_part
    output[starting_lum-3:starting_lum+n_shells-4] = lum_part
    return output

def unstitch_vector(reduced_state_vector, starting_indices):
    """
        Given a reduced state vector, insert 0's in the appropriate places so that the dimension matches the state vector of the system
        Input:
            reduced_state_vector: 4(k-1) dimensional array where k is the number of mass shells
            starting_indices: The starting index of each variable for the state vector, NOT the reduced_state_vector (use gen_starting_index to produce)
        Output:
            out_vec: 4k dimensional array, with 0's in appropriate places
    """
    dim = reduced_state_vector.shape[0]
    assert(dim != 0)
    assert(dim%4==0)
    n_shells_minus_one = int(dim/4)
    n_shells = n_shells_minus_one+1
    output = np.zeros(4*n_shells)


    starting_rad = starting_indices[StateVectorVar.RADIUS]
    starting_p = starting_indices[StateVectorVar.PRESSURE]
    starting_temp = starting_indices[StateVectorVar.TEMP]
    starting_lum = starting_indices[StateVectorVar.LUMINOSITY]

    # Include additional offset to account for the fact that starting_indices is for original index
    starting_rad_reduced = starting_rad
    starting_p_reduced = starting_p-1
    starting_temp_reduced = starting_temp-2
    starting_lum_reduced = starting_lum-3

    rad_part = reduced_state_vector[starting_rad_reduced: starting_rad_reduced+n_shells_minus_one]
    p_part = reduced_state_vector[starting_p_reduced: starting_p_reduced+n_shells_minus_one]
    t_part = reduced_state_vector[starting_temp_reduced: starting_temp_reduced+n_shells_minus_one]
    lum_part = reduced_state_vector[starting_lum_reduced: starting_lum_reduced+n_shells_minus_one]

    output[starting_rad+1: starting_rad+1+n_shells_minus_one] = rad_part # Don't touch r0
    output[starting_p: starting_p+n_shells_minus_one] = p_part # Don't touch p_{k-1}
    output[starting_temp: starting_temp+n_shells_minus_one] = t_part # Don't touch T_{k-1}
    output[starting_lum+1: starting_lum+1+n_shells_minus_one] = lum_part# Don't touch r0

    return output
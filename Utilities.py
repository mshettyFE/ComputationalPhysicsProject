import numpy as np

# values of fundamental constants in SI units
G = np.float64(6.6743E-11) # (Nm^{2})/(kg^{2})
StefanBoltz = np.float64(5.67E-8) # W/(m^{2}K^{4})
Boltzman = np.float64(1.380649E-23) # J/K
m_p = np.float64(1.67262192E-27)

# Parameters for the Sun
M_sun = 1.989E30 # kg
R_sun  = 6.9634E8 # m
mu_sun = np.float64(0.6)

# Numerical Resolution used throughout the sim
global_tolerance = 1E-9

MASS_UNIT_INDEX = 0
RADIUS_UNIT_INDEX = 1
DENSITY_UNIT_INDEX = 2
PRESSURE_UNIT_INDEX = 3
LUMINOSITY_UNIT_INDEX = 4
TEMP_UNIT_INDEX = 5
TIME_UNIT_INDEX = 6

def UnitScalingFactors(R_0, M_0):
    """
    Returns the scaling factors to convert the unitless 
    numbers in the sim to physical units.
    Inputs:
        R_0: Length Scale
        M_0: Mass Scale
    Output:
        6x1 numpy array whose elements are the scaling factors.
        Use the *_UNIT_INDEX variables to get the corresponding scalings
    """
    assert(R_0 > 0)
    assert(M_0 > 0)
    R_out = np.float64(R_0)
    M_out = np.float64(M_0)
    rho_out = M_0/(np.power(R_0,3))
    t_0 = np.sqrt(np.power(R_0,3) / (G*M_0))
    P_out = M_0/(R_0*t_0*t_0)
    L_out = (M_0*np.power(R_0,2)) / (np.power(t_0,3))
    T_out = (M_0*np.power(R_0,2)) / (t_0*t_0*Boltzman)
    return np.array([
            M_out,
            R_out,
            rho_out,
           P_out,
           L_out,
           T_out
        ])

def generate_extra_parameters(R_0, M_0, epsilon, kappa, mu):
    """
        Given the unitful parameters of the problem, generate the unitless constants to be used in the simulation
    Inputs:
        R_0: Length Scale
        M_0: Mass Scale
        epsilon: tuning parameter for luminosity equation
        kappa: tuning parameter for temperature equation
        mu: mean molecular weight in units of proton mass
    Output:
        extra_const_params: python dictionary containing the converted constant parameters
    """
    scale_factors = UnitScalingFactors(R_0, M_0)
    t_0 = np.sqrt(np.power(R_0,3) / (G*M_0))
    T_0 = scale_factors[TEMP_UNIT_INDEX]
    new_ep = epsilon * np.power(t_0,3)*M_0*np.power(T_0,4)* np.power(R_0,3)
    new_kp = kappa* (3/16*StefanBoltz)* M_0/ (np.power(R_0,5)*np.power( T_0,7.5)* np.power(t_0,3))
    extra_const_params = {
        "mu": mu,
        "m_p_prime": m_p/M_0,
    "epsilon_prime": new_ep,
    "kappa_prime": new_kp,
    }

    return extra_const_params

def equation_of_state(P_prime, T_prime, extra_const_params):
    """
        generate the density given the pressure and temperature
        Input:
            P_prime: dimensionless pressure (np.float64)
            T_prime: dimensionless temp (np.float64)
            extra_const_params: 
        Output:
            rho_prime: dimensionless density  (np.float64)
    """
    rho_prime = (P_prime*extra_const_params["mu"]* extra_const_params["m_p_prime"])/T_prime
    return rho_prime
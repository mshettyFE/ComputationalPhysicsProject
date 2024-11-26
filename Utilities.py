import numpy as np

# values of fundamental constants in SI units
G = np.float64(6.6743E-11) # (Nm^{2})/(kg^{2})
StefanBoltz = np.float64(5.67E-8) # W/(m^{2}K^{4})
Boltzman = np.float64(1.380649E-23) # J/K
m_p = np.float64(1.67262192E-27) # kg

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


def UnitScalingFactors( M_0, R_0):
    """
    Returns the scaling factors to convert the unitless 
    numbers in the sim to physical units.
    Inputs:
        M_0: Mass Scale
        R_0: Length Scale
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
    P_out = M_0/(R_0*np.power(t_0,2))
    L_out = (M_0*np.power(R_0,2)) / (np.power(t_0,3))
    T_out = (m_p*np.power(R_0,2)) / (np.power(t_0,2)*Boltzman) # Using m_p here since T_out is the temp scale on a particle level, not on a star level
    return np.array([
            M_out,
            R_out,
            rho_out,
           P_out,
           L_out,
           T_out,
           t_0
        ])


def generate_extra_parameters(M_0, R_0, E_0, kappa, mu):
    """
        Given the unitful parameters of the problem, generate the unitless constants to be used in the simulation
    Inputs:
        R_0: Length Scale
        M_0: Mass Scale
        epsilon: nuclear energy generation constant for luminosity equation [J*m^3/kg^2/s]
        kappa: opacity parameter for temperature equation [m^2/kg]
        mu: mean molecular weight in units of proton mass
    Output:
        extra_const_params: python dictionary containing the converted constant parameters
    """
    scale_factors = UnitScalingFactors(M_0, R_0)
    t_0 = np.sqrt(np.power(R_0,3) / (G*M_0))
    T_0 = scale_factors[TEMP_UNIT_INDEX]
    new_ep = E_0/(scale_factors[DENSITY_UNIT_INDEX]*np.power(T_0,4))*(M_0/scale_factors[LUMINOSITY_UNIT_INDEX])
#    new_ep = E_0* np.power(t_0,3)*M_0*np.power(T_0,4)/ np.power(R_0,5) #E_0 is dependent on the main reactants (proton-proton fusion for the Sun)
    new_kp = kappa* (3/(16*StefanBoltz)) * np.power(M_0,3)/ (np.power(R_0,5)*np.power( T_0,7.5)* np.power(t_0,3)) #kappa_0 is dependent on the region of interest and varies from 0.02 (core) to 0.001 (exterior)
    
    extra_const_params = {
        "mu": mu,
    "E_prime": new_ep,
    "kappa_prime": new_kp,
#    "E_prime": E_0,
#    "kappa_prime": kappa,
    } 

    return extra_const_params



def equation_of_state(P_prime, T_prime, extra_const_params):
    """
        generate the density given the pressure and temperature
        Input:
            P_prime: dimensionless pressure (np.float64)
            T_prime: dimensionless temp (np.float64)
            extra_const_params: constant parameters.
        Output:
            rho_prime: dimensionless density  (np.float64)
    """
    rho_prime = (P_prime*extra_const_params["mu"])/T_prime
    return rho_prime



def nuclear_energy(rho_prime, T_prime, extra_const_params):
    """
        generate the nuclear energy production #rate? given the density and temperature
        Input:
            rho_prime: dimensionless density (np.float64)
            T_prime: dimensionless temp (np.float64)
        Output:
            E_prime: dimensionless energy #rate  (np.float64)
    """
    E_prime = (extra_const_params["E_prime"])*rho_prime**(1)*T_prime**(4)
    return E_prime
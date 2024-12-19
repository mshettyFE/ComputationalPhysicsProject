import jax.numpy as jnp
from enum import Enum

# values of fundamental constants in SI units
G = jnp.float64(6.6743E-11) # (Nm^{2})/(kg^{2})
StefanBoltz = jnp.float64(5.67E-8) # W/(m^{2}K^{4})
Boltzman = jnp.float64(1.380649E-23) # J/K
m_p = jnp.float64(1.67262192E-27) # kg

# Parameters for the Sun
M_sun = 1.989E30 # kg
R_sun  = 6.9634E8 # m
mu_sun = jnp.float64(0.6)
E_0_sun = 1.8E-29 # m^5/(kg s^3*K^4)
kappa_0_sun = 3E-2 # m^2/kg

class ScaleIndex(Enum):
    RADIUS = 0
    PRESSURE = 1
    TEMP = 2
    LUMINOSITY = 3
    MASS = 4
    DENSITY= 5
    TIME = 6

def UnitScalingFactors( M_0, R_0):
    """
    Returns the scaling factors to convert the unitless 
    numbers in the sim to physical units.
    Inputs:
        M_0: Mass Scale
        R_0: Length Scale
    Output:
        Dictionary of {ScaleIndex: jnp.float64}, where ScaleIndex denotes which unit, and jnp.float64 denotes scale factor
        Use the *_UNIT_INDEX variables to get the corresponding scalings
    """
    assert(R_0 > 0)
    assert(M_0 > 0)
    #The "out" variables are the coefficients which multiply the scaled variables and generate the original unit variables.
    R_out = jnp.float64(R_0)
    M_out = jnp.float64(M_0)
    rho_out = M_0/(jnp.power(R_0,3))
    t_0 = jnp.sqrt(jnp.power(R_0,3) / (G*M_0))
    P_out = M_0/(R_0*jnp.power(t_0,2))
    L_out = (M_0*jnp.power(R_0,2)) / jnp.power(t_0,3)
    T_out = (m_p*jnp.power(R_0,2)) / (jnp.power(t_0,2)*Boltzman) # Using m_p here since T_out is the temp scale on a particle level, not on a star level
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

    ep_prefactor = jnp.power(R_0,5)*jnp.power(M_0,-1)*jnp.power(T0,-4)*jnp.power(t0,-3)
    new_ep = E_0/ep_prefactor


    kp_prefactor = jnp.power(R_0,5)*jnp.power(T0,3.5)/jnp.power(M_0,2)
    scaled_stefan = StefanBoltz*(jnp.power(R_0,2)*jnp.power(T0,4))/L0
    kp_const_prefactor = (3/(16*scaled_stefan*jnp.power(4*jnp.pi,2) ))
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
    E_prime = (extra_const_params["E0_prime"])*rho_prime*jnp.power(T_prime,4)
    return E_prime
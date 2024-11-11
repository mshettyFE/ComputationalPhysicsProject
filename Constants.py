import numpy as np

# values of fundamental constants in SI units
G = np.float64(6.6743E-11) # (Nm^{2})/(kg^{2})
StefanBoltz = np.float64(5.67E-8) # W/(m^{2}K^{4})
Boltzman = np.float64(1.380649E-23) # J/K

# The epxected dimensionality of the problem at hand
dimensionality = (6,1)

MASS_UNIT_INDEX = 0
RADIUS_UNIT_INDEX = 1
DENSITY_UNIT_INDEX = 2
PRESSURE_UNIT_INDEX = 3
LUMINOSITY_UNIT_INDEX = 4
TEMP_UNIT_INDEX = 5

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
    T_out = (M_0*np.power(R_0,2)) / (t_0*t_0*k)
    return np.array([
            M_out,
            R_out,
            rho_out,
           P_out,
           L_out,
           T_out
        ])



import numpy as np
from StateVector import InterpolationIndex, StateVector

# NOTE: x_{half} = 0.5(x_{k+1}-x_{k})
# so we see that (x_{k+1}-x_{k}) = 2*x_{half}

# Difference equation for radius
# (r_{k+1}-r_{k})/(dm)- 1/(4*pi*r^{2}_{half}* \rho_{half})
def calc_gr(interpolation, dm):
    rad = interpolation[InterpolationIndex.RADIUS.value,:]
    density = interpolation[InterpolationIndex.DENSITY.value,:]
    return rad*2/dm-1/(4*np.pi)/np.power(rad,2)/density

# Difference equation for Pressure
# (P_{k+1}-P_{k})/(dm) +  (dm/2)/(4*pi*r^{4}_{half})
def calc_gP(interpolation, dm):
    pres = interpolation[InterpolationIndex.PRESSURE.value, :]
    rad = interpolation[InterpolationIndex.RADIUS.value,:]
    return pres*2/dm+dm/2/(4*np.pi*np.power(rad,4))

# Difference equation for Temperature
# (T_{k+1}-T_{k})/(dm)+ \kappa_0 \rho_{half}* L_{half}/r^{4}_{half}/T^{6.5}_{half}
def calc_gT(interpolation, dm, kappa):
    temp = interpolation[InterpolationIndex.TEMP.value, :]
    density =  interpolation[InterpolationIndex.DENSITY.value,:]
    lum = interpolation[InterpolationIndex.LUMINOSITY.value, :]
    rad = interpolation[InterpolationIndex.RADIUS.value,:]
    return temp*2/dm+kappa*density*lum/np.power(rad,4)/np.power(temp,6.5)

# Difference equation for luminosity
# (L_{k+1}-L_{k})/(dm)- \epsilon_0 \rho_{half}*T_{half}^{4}
def calc_gL(interpolation, dm, epsilon):
    temp = interpolation[InterpolationIndex.TEMP.value, :]
    density =  interpolation[InterpolationIndex.DENSITY.value,:]
    lum = interpolation[InterpolationIndex.LUMINOSITY.value, :]
    return lum*2/dm-epsilon*density*np.power(temp,4)    

def calc_g(state_vector: StateVector, parameters):
    interpolation = state_vector.interpolate_all(parameters)
    dm = 1/state_vector.n_shells
    gr = calc_gr(interpolation, dm)
    gP = calc_gP(interpolation, dm)
    gT = calc_gT(interpolation, dm, parameters["k0_prime"])
    gL = calc_gL(interpolation, dm, parameters["E0_prime"])
    return np.concatenate([gr,gP,gT, gL], axis=None)
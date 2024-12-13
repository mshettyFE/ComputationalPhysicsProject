import numpy as np
from StateVector import InterpolationIndex, StateVector, StateVectorVar

# NOTE: x_{half} = 0.5(x_{k+1}-x_{k})
# so we see that (x_{k+1}-x_{k}) = 2*x_{half}

# Difference equation for radius
# (r_{k+1}-r_{k})/(dm)- 1/(4*pi*r^{2}_{half}* \rho_{half})
def calc_gr(differences, interpolation, dm):
    rad = differences[StateVectorVar.RADIUS.value,:]
    interp_rad =  interpolation[InterpolationIndex.RADIUS.value,:]
    density = interpolation[InterpolationIndex.DENSITY.value,:]
    return rad*2/dm-1/(4*np.pi)/np.power(interp_rad,2)/density

# Difference equation for Pressure
# (P_{k+1}-P_{k})/(dm) +  (dm/2)/(4*pi*r^{4}_{half})
def calc_gP(differences,interpolation, dm):
    pres = differences[StateVectorVar.PRESSURE.value, :]
    rad = interpolation[InterpolationIndex.RADIUS.value,:]
    return pres*2/dm+dm/2/(4*np.pi*np.power(rad,4))

# Difference equation for Temperature
# (T_{k+1}-T_{k})/(dm)+ \kappa_0 \rho_{half}* L_{half}/r^{4}_{half}/T^{6.5}_{half}
def calc_gT(differences , interpolation, dm, kappa):
    temp = differences[StateVectorVar.TEMP.value, :]
    interp_temp = interpolation[InterpolationIndex.TEMP.value,:] 
    density =  interpolation[InterpolationIndex.DENSITY.value,:]
    lum = interpolation[InterpolationIndex.LUMINOSITY.value, :]
    rad = interpolation[InterpolationIndex.RADIUS.value,:]
    return temp*2/dm+kappa*density*lum/np.power(rad,4)/np.power(interp_temp,6.5)

# Difference equation for luminosity
# (L_{k+1}-L_{k})/(dm)- \epsilon_0 \rho_{half}*T_{half}^{4}
def calc_gL(differences, interpolation, dm, epsilon):
    temp = interpolation[InterpolationIndex.TEMP.value, :]
    density =  interpolation[InterpolationIndex.DENSITY.value,:]
    lum = differences[StateVectorVar.LUMINOSITY.value, :]
    return lum*2/dm-epsilon*density*np.power(temp,4)    

def calc_g(state_vector: StateVector, parameters):
    interpolation = state_vector.interpolate_all(parameters)
    diffs = state_vector.diff_vars_all()
    dm = 1/state_vector.n_shells
    block_size = state_vector.n_shells-1
    output = np.zeros(4*block_size)
    output[0:block_size] = calc_gr(diffs,interpolation, dm)
    output[0:block_size] = calc_gP(diffs,interpolation, dm)
    output[0:block_size] = calc_gT(diffs,interpolation, dm, parameters["k0_prime"])
    output[0:block_size] = calc_gL(diffs,interpolation, dm, parameters["E0_prime"])
    return output
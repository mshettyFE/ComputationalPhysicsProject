import numpy as np

# NOTE: x_{half} = 0.5(x_{k+1}-x_{k})
# so we see that (x_{k+1}-x_{k})*2 = x_{half}

# Difference equation for radius
# (r_{k+1}-r_{k})/(dm)- 1/(4*pi*r^{2}_{half}* \rho_{half})
def calc_gr(radius_interp, density_interp, dm):
    pass

# Difference equation for Pressure
# (P_{k+1}-P_{k})/(dm) +  (dm/2)/(4*pi*r^{4}_{half})
def calc_gP(Pressure_interp, radius_interp, dm):
    pass

# Difference equation for Temperature
# (T_{k+1}-T_{k})/(dm)+ \kappa_0 \rho_{half}* L_{half}/r^{4}_{half}/T^{6.5}_{half}
def calc_gT(Temperature_interp, density_interp, lum_interp, radius_interp, dm, kappa):
    pass

# Difference equation for luminosity
# (L_{k+1}-L_{k})/(dm)- \epsilon_0 \rho_{half}*T_{half}^{4}
def calc_gL(Luminosity_interp, density_interp, temp_interp):
    pass

def calc_g(state_vector, starting_indices, parameters):
    dim = state_vector.shape[0]
    assert(dim != 0)
    assert(dim%4==0)
    n_shells = int(dim/4)
    output_dim = 4*(n_shells-1)
    return np.zeros(output_dim, output_dim)
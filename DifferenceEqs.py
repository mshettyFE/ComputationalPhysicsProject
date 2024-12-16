import jax.numpy as jnp
from StateVector import InterpolationIndex, StateVector, StateVectorVar
import Utilities

# dm = m_{k+1}-m_{k}

def calc_jax_g(state_vector, indicies, n_shells, dm, constants):

    starting_rad_index = indicies[StateVectorVar.RADIUS]
    r_k = state_vector[starting_rad_index:starting_rad_index+n_shells-1]
    r_k_1 = state_vector[starting_rad_index+1:starting_rad_index+n_shells]

    starting_pres_index = indicies[StateVectorVar.PRESSURE]
    p_k = state_vector[starting_pres_index:starting_pres_index+n_shells-1]
    p_k_1 = state_vector[starting_pres_index+1:starting_pres_index+n_shells]

    starting_temp_index = indicies[StateVectorVar.TEMP]
    t_k = state_vector[starting_temp_index:starting_temp_index+n_shells-1]
    t_k_1 = state_vector[starting_temp_index+1:starting_temp_index+n_shells]

    starting_lum_index = indicies[StateVectorVar.LUMINOSITY]
    l_k = state_vector[starting_lum_index:starting_lum_index+n_shells-1]
    l_k_1 = state_vector[starting_lum_index+1:starting_lum_index+n_shells]

    interp_rad = (r_k+r_k_1)/2
    dif_rad = (r_k_1-r_k)

    interp_pres = (p_k+p_k_1)/2
    dif_pres = (p_k_1-p_k)

    interp_temp = (t_k+t_k_1)/2
    dif_temp = (t_k_1-t_k)

    interp_lum = (l_k+l_k_1)/2
    dif_lum = (l_k_1-l_k)

    density = Utilities.equation_of_state(interp_pres, interp_temp, constants)
    shell_masses = dm*jnp.arange(n_shells)
    masses = 0.5*(shell_masses[0:n_shells-1]+shell_masses[1:n_shells])
    output = jnp.zeros(4*n_shells)

# Difference equation for radius
# (r_{k+1}-r_{k})/(dm)- 1/(4*pi*r^{2}_{half}* \rho_{half})
    radial = dif_rad/dm-1/(4*jnp.pi)/jnp.power(interp_rad,2)/density
    output = output.at[starting_rad_index+1:starting_rad_index+n_shells].add(radial) # ignore r_0 term
# Difference equation for Pressure
# (P_{k+1}-P_{k})/(dm) +  (m_{half})/(4*pi*r^{4}_{half})
    pressure=  dif_pres/dm+masses/(4*jnp.pi*jnp.power(interp_rad,4))
    output = output.at[starting_pres_index:starting_pres_index+n_shells-1].add(pressure)# ignore P_k-1 term

# Difference equation for Temperature
# (T_{k+1}-T_{k})/(dm)+ \kappa_0 \rho_{half}* L_{half}/r^{4}_{half}/T^{6.5}_{half}
    temperature = dif_temp/dm+constants["k0_prime"]*density*interp_lum/jnp.power(interp_rad,4)/jnp.power(interp_temp,6.5)
    output = output.at[starting_temp_index:starting_temp_index+n_shells-1].add(temperature)# ignore T_k-1 term

# Difference equation for luminosity
# (L_{k+1}-L_{k})/(dm)- \epsilon_0 \rho_{half}*T_{half}^{4}
    lum = dif_lum/dm-constants["E0_prime"]*density*jnp.power(interp_temp,4) # ignore L_0 term
    output = output.at[starting_lum_index+1:starting_lum_index+n_shells].add(lum)
    return output
import jax.numpy as jnp
from StateVector import InterpolationIndex, StateVector, StateVectorVar
import Utilities

# dm = m_{k+1}-m_{k}

def calc_jax_g_log(state_vector, indicies, n_shells, dm, constants):
    """
        Set up residual function to be compatible with Jax, but assumes state_vector has log variables
        Inputs:
            state_vector: 4k dimensional array of values
            indicies: Starting index of each variable in state_vector
            n_shells: number of shells in state_vector
            dm: mass assigned to each shell
            constants: Dictionary of external parameters
        Output:
            output: 4*n_shells containing the residuals for each shell. Boundary conditions hard coded as having 0 residual
    """
    # Extract variables
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

    # Convert back to linear space and calculate interpolated and difference variables
    exp_r_k = jnp.exp(r_k)
    exp_r_k_1 = jnp.exp(r_k_1)
    exp_p_k = jnp.exp(p_k)
    exp_p_k_1 = jnp.exp(p_k_1)
    exp_t_k = jnp.exp(t_k)
    exp_t_k_1 = jnp.exp(t_k_1)
    exp_l_k = jnp.exp(l_k)
    exp_l_k_1 = jnp.exp(l_k_1)

    interp_rad = (exp_r_k+exp_r_k_1)/2
    dif_rad = (exp_r_k_1-exp_r_k)

    interp_pres = (exp_p_k+exp_p_k_1)/2
    dif_pres = (exp_p_k_1-exp_p_k)

    interp_temp = (exp_t_k+exp_t_k_1)/2
    dif_temp = (exp_t_k_1-exp_t_k)

    interp_lum = (exp_l_k+exp_l_k_1)/2
    dif_lum = (exp_l_k_1-exp_l_k)

    density = Utilities.equation_of_state(interp_pres, interp_temp, constants)
    shell_masses = dm*jnp.arange(n_shells)
    masses = 0.5*(shell_masses[0:n_shells-1]+shell_masses[1:n_shells])
    output = jnp.zeros(4*n_shells)

    # Same calculation as calc_jax_g
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

#    print("NAN")
#    print(jnp.isnan(radial).any())
#    print(jnp.isnan(pressure).any())
#    print(jnp.isnan(constants["k0_prime"]*density*interp_lum/jnp.power(interp_rad,4)/jnp.power(interp_temp,6.5)).any())
#    print(jnp.isnan(lum).any())
    return output


def calc_jax_g(state_vector, indicies, n_shells, dm, constants):
    """
        Set up residual function to be compatible with Jax
        Inputs:
            state_vector: 4k dimensional array of values
            indicies: Starting index of each variable in state_vector
            n_shells: number of shells in state_vector
            dm: mass assigned to each shell
            constants: Dictionary of external parameters
        Output:
            output: 4*n_shells containing the residuals for each shell. Boundary conditions hard coded as having 0 residual
    """

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
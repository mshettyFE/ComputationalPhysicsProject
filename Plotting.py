import jax.numpy as jnp
from StateVector import StateVector, StateVectorVar, DataGenMode
import matplotlib.pyplot as plt
import Utilities

def plot_save(x,y, fname, title, x_axis, y_axis, log_x=False, log_y=False):
    plt.clf()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if(log_x):
        plt.xscale("log")
    if (log_y):
        plt.yscale("log")
    plt.savefig(fname)

if __name__ == "__main__":
    s =  StateVector(50, DataGenMode.TEST)
    s.load_state("REPLACEME.npy")
    print(s.state_vec.shape)
    r_start = s.starting_indices[StateVectorVar.RADIUS]
    l_start  =s.starting_indices[StateVectorVar.LUMINOSITY]
    p_start  =s.starting_indices[StateVectorVar.PRESSURE]
    t_start  =s.starting_indices[StateVectorVar.TEMP]
    params = Utilities.generate_extra_parameters(Utilities.M_sun, Utilities.R_sun, Utilities.E_0_sun, Utilities.kappa_0_sun, Utilities.mu_sun)
    density = Utilities.equation_of_state(s.state_vec[p_start:p_start+s.n_shells],s.state_vec[t_start:t_start+s.n_shells], params)
    radius = s.state_vec[r_start:r_start+s.n_shells]
    lum = s.state_vec[l_start:l_start+s.n_shells]
    pres = s.state_vec[p_start:p_start+s.n_shells] 
    temper =  s.state_vec[t_start:t_start+s.n_shells]
    plot_save(radius, lum, "LumRad.png", "Luminosity v Radius", "Radius", "Luminosity")
    plot_save(radius, pres, "PressureRad.png", "Pressure v Rad", "Radius", "Pressure")
    plot_save(radius, temper, "TempRad.png", "Temp v Rad", "Radius", "Temperature")
    plot_save(radius, density, "DensityRad.png", "Density v Rad", "Radius", "Density")
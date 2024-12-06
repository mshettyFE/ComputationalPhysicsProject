import numpy as np
import matplotlib.pyplot as plt
import Minimizer
import Integrator
from Utilities import * #maybe not eeeeverything
from mpl_toolkits import mplot3d

if __name__ == "__main__":
    
    m_steps = 1000 #Number of mass steps.
    #Initialization of multiple stars.

    #!!!        M_star and R_star (also maybe E_0 &  kappa_0) need to be normalized beforethey are plugged into anything.
    M_star = np.array([100,26.9,30,36,40,45,95,58,118]) #Array of stellar masses ranging from 1sol to 100sol.
    R_star = np.array([92,8.3,76,103,25,200,20,86,22.3]) #Array of stellar radii corresponding to each M_star
    E_0 = np.array([E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun]) #Array of stellar energy rate constants corresponding to each M_star
    kappa_0 = np.array([kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun]) #Array of stellar opacity constants corresponding to each M_star
    mu = np.array([mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun]) #Array of stellar molecular weight corresponding to each M_star
    MRL_Mapping = np.zeros((len(mu),3)) #Initilization.

    for i in range(len(mu)):
        #Calls the necessary functions to run the program and generate the state array len(mu) times
        #corresponding to len(mu) total data points.
        scale_factors = UnitScalingFactors(M_star[i], R_star[i])
        constants = generate_extra_parameters(M_star[i], R_star[i], E_0[i], kappa_0[i], mu[i])

        #!!!     Guesses need to scale with each radius.
        PT_ideal = Minimizer.run_minimizer(2E7/scale_factors[TEMP_UNIT_INDEX],30E15/scale_factors[PRESSURE_UNIT_INDEX],
            m_steps, M_star[i], R_star[i], constants["E_prime"], constants["kappa_prime"], mu[i])
        init_conds = Minimizer.gen_initial_conditions(PT_ideal.x[0], PT_ideal.x[1],1E-2, constants)
        state_array = Integrator.ODESolver(init_conds, m_steps, constants)
        #Preparing the plot by creating an array with the coordinates for all data points:
        MRL_Mapping[i, :] = [M_star[i], R_star[i], state_array[m_steps-1, LUMINOSITY_UNIT_INDEX]]

    #Generates scatter plot on 3D graph.
    fig = plt.figure()
    plt.title('Total Stellar Mass, Radius, and Luminosity')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Mass")
    ax.set_ylabel("Radius")
    ax.set_zlabel("Luminosity")
    
    #Actual data:
    MRL_Data = np.array([
    [100, 92, 3.2E6],       # WR102KA
    [26.9, 8.27, 1.02E5],   # 10Lacertae
    [30, 76, 6.1E5],        # PCygni
    [36, 103, 8.5E5],       # ZetaScorpii
    [40, 25, 3.39E5],       # MuNormae
    [45, 200, 1.26E6],      # HD50064
    [95, 20, 2.5E6],        # HD93129A
    [58, 86, 2.5E6],        # WR102EA
    [118, 22.3, 2E6]])      # R136B

    
    # Plot computed data points
    ax.scatter(MRL_Mapping[:, 0], MRL_Mapping[:, 1], MRL_Mapping[:, 2], c='blue', marker='o', label='Computed Data')

    # Add dashed lines for computed data points
    for i in range(MRL_Mapping.shape[0]):
        ax.plot(
        [MRL_Mapping[i, 0], MRL_Mapping[i, 0]],  # Fixed Mass (X-axis)
        [MRL_Mapping[i, 1], MRL_Mapping[i, 1]],  # Fixed Radius (Y-axis)
        [0, MRL_Mapping[i, 2]],  # From Z=0 to Z=Luminosity
        linestyle='--', color='blue', alpha=0.5
                )

    # Plot actual data points
    ax.scatter(MRL_Data[:, 0], MRL_Data[:, 1], MRL_Data[:, 2], c='red', marker='o', label='Actual Data')

    # Add dashed lines for actual data points
    for i in range(MRL_Data.shape[0]):
        ax.plot(
            [MRL_Data[i, 0], MRL_Data[i, 0]],  # Fixed Mass (X-axis)
            [MRL_Data[i, 1], MRL_Data[i, 1]],  # Fixed Radius (Y-axis)
            [0, MRL_Data[i, 2]],  # From Z=0 to Z=Luminosity
            linestyle='--', color='red', alpha=0.5
                )

    # Add legend
    ax.legend()

    # Save and display the plot
    plt.savefig('MRL')

  
   
    

                    


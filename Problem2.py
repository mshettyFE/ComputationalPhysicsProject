import numpy as np
import matplotlib.pyplot as plt
import Minimizer
import Integrator
from Utilities import * #maybe not eeeeverything
from mpl_toolkits import mplot3d

if __name__ == "__main__":
    
    m_steps = 1000 #Number of mass steps.
    #Initialization of multiple stars.

    #!!!        E_0 &  kappa_0 need to be normalized before they are plugged into anything       ???
    M_star = np.array([100,26.9,30,36,40,45,95,58,118])*M_sun #Array of stellar masses ranging from 1sol to 100sol.
    R_star = np.array([92,8.3,76,103,25,200,20,86,22.3])*R_sun #Array of stellar radii corresponding to each M_star
    E_0 = np.array([E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun,E_0_sun]) #Array of stellar energy rate constants corresponding to each M_star
    kappa_0 = np.array([kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun,kappa_0_sun]) #Array of stellar opacity constants corresponding to each M_star
    mu = np.array([mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun,mu_sun]) #Array of stellar molecular weight corresponding to each M_star
    MRL_Mapping = np.zeros((len(mu),3)) #Initilization.

    for i in range(len(mu)):
        #Calls the necessary functions to run the program and generate the state array len(mu) times
        #corresponding to len(mu) total data points/stars.
        scale_factors = UnitScalingFactors(M_star[i], R_star[i]) #Scales parameters to unitless, ~ order(1).
        constants = generate_extra_parameters(M_star[i], R_star[i], E_0[i], kappa_0[i], mu[i]) #Returns E_0_prime, kappa_0_prime, mu_prime which is grouped with other constants.

        PT_ideal = Minimizer.run_minimizer(1E7/scale_factors[TEMP_UNIT_INDEX],1E16/scale_factors[PRESSURE_UNIT_INDEX], #Initial guess of temp[K] and pressure[Pa] of core is average enough to apply to the mass range of the sample stars.
            m_steps, M_star[i], R_star[i], constants["E_prime"], constants["kappa_prime"], mu[i]) #Output is minimized unitless P and T.
        init_conds = Minimizer.gen_initial_conditions(PT_ideal.x[0], PT_ideal.x[1],1E-2, constants) #Output is the other set of initial unitless parameters based on minimized P and T above.
        state_array = Integrator.ODESolver(init_conds, m_steps, constants) #State array of parameters at each mass step. 
        #Preparing the plot by creating an array with the coordinates for all data points:
        MRL_Mapping[i, :] = [M_star[i], state_array[-1, RADIUS_UNIT_INDEX], state_array[-1, LUMINOSITY_UNIT_INDEX]]

    print(MRL_Mapping)
    #Generates scatter plot on 3D graph.
    fig, ax1 = plt.subplots()
    
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
    plt.title('Total Stellar Mass versus Radius')
    ax1.set_xlabel("Mass")
    ax1.set_ylabel("Rad")
    ax1.scatter(MRL_Mapping[:, 0], MRL_Mapping[:, 1], c='blue', marker='o', label='Computed Data')
    ax1.legend()
    plt.savefig('MR')

    ax1.clear()
    plt.title('Total Stellar Mass versus Radius')
    ax1.set_xlabel("Mass")
    ax1.set_ylabel("Rad")
    ax1.scatter(MRL_Mapping[:, 0], MRL_Mapping[:, 2], c='blue', marker='o', label='Computed Data')
    ax1.legend()
    plt.savefig('ML')

    # Add dashed lines for computed data points
#    for i in range(MRL_Mapping.shape[0]):
#        ax.plot(
#        [MRL_Mapping[i, 0], MRL_Mapping[i, 0]],  # Fixed Mass (X-axis)
#        [MRL_Mapping[i, 1], MRL_Mapping[i, 1]],  # Fixed Radius (Y-axis)
#        [0, MRL_Mapping[i, 2]],  # From Z=0 to Z=Luminosity
#        linestyle='--', color='blue', alpha=0.5
#                )

    # Plot actual data points
    #ax.scatter(MRL_Data[:, 0], MRL_Data[:, 1], MRL_Data[:, 2], c='red', marker='o', label='Actual Data')

    # Add dashed lines for actual data points
#    for i in range(MRL_Data.shape[0]):
#        ax.plot(
#            [MRL_Data[i, 0], MRL_Data[i, 0]],  # Fixed Mass (X-axis)
#            [MRL_Data[i, 1], MRL_Data[i, 1]],  # Fixed Radius (Y-axis)
#            [0, MRL_Data[i, 2]],  # From Z=0 to Z=Luminosity
#            linestyle='--', color='red', alpha=0.5
#                )
  
   
    

                    


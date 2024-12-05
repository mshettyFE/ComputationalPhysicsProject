import numpy as np
import matplotlib.pyplot as plt
import Minimizer
import Integrator
from Utilities import * #maybe not eeeeverything
from mpl_toolkits import mplot3d

if __name__ == "__main__":
    
    m_steps = 1000
    R_star = np.array([])
    M_star = np.array([])
    E_0 = np.array([])
    kappa_0 = np.array([])
    mu = np.array([])
    MRL_Mapping = np.zeros((len(mu),3))

    for i in range(len(mu)):
        #Calls the necessary functions to run the program and generate the state array len(mu) times
        #corresponding to len(mu) total data points.
        scale_factors = UnitScalingFactors(M_star[i], R_star[i])
        constants = generate_extra_parameters(M_star[i], R_star[i], E_0[i], kappa_0[i], mu[i])
        results = Minimizer.run_minimizer(2E7/scale_factors[TEMP_UNIT_INDEX],30E15/scale_factors[PRESSURE_UNIT_INDEX],
            m_steps, M_star[i], R_star[i], constants["E_prime"], constants["kappa_prime"], mu[i])
        init_conds = Minimizer.gen_initial_conditions(results.x[0], results.x[1],1E-2, constants)
        state_array = Integrator.ODESolver(init_conds, m_steps, constants)
        #Preparing the plot by creating an array with the coordinates for all data points:
        MRL_Mapping[i, :] = [M_star[i], R_star[i], state_array[LUMINOSITY_UNIT_INDEX, m_steps]]


    #Generates scatter plot on 3D graph.
    fig = plt.figure()
    plt.title('Total Mass Effect on Stellar Radius and Luminosity')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Mass")
    ax.set_ylabel("Radius")
    ax.set_zlabel("Luminosity")
    ax.scatter(MRL_Mapping[:, 0], MRL_Mapping[:, 1], MRL_Mapping[:, 2], c='blue', marker='o')
    plt.savefig('MRL')

  
   
    

                    


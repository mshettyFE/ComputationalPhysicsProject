import numpy as np
from matplotlib import pyplot as plt
import Integrator
from Utilities import *

if __name__ == "__main__":  # Main guard ensures this code runs only when the script is executed directly
    pass

#MASS_UNIT_INDEX = 0 []
#RADIUS_UNIT_INDEX = 1 []
#DENSITY_UNIT_INDEX = 2 []
#PRESSURE_UNIT_INDEX = 3 [dynes/cm^2]
#LUMINOSITY_UNIT_INDEX = 4 []
#TEMP_UNIT_INDEX = 5 [K]
#TIME_UNIT_INDEX = 6 

eps = 1e-6 #parameter to prevent divide by zero
# Generate multi-star states with 3 sets of initial conditions. INPUT INITIAL CONDITIONS.
state0 = Integrator.ODESolver([0,eps,eps,2.5E14,0,1.5E7], 100, generate_extra_parameters(M_sun, R_sun, 1.4E5, 0.2, mu_sun))  
state1 = Integrator.ODESolver([0,eps,eps,2.5E14,0,1.5E7], 100, generate_extra_parameters(M_sun, R_sun, 1.4E5, 0.2, mu_sun))  
state2 = Integrator.ODESolver([0,eps,eps,2.5E14,0,1.5E7], 100, generate_extra_parameters(M_sun, R_sun, 1.4E5, 0.2, mu_sun))  

MSS = np.stack([state0, state1, state2], axis=2) #7x(step-size)x3 array. MSS = Multi-Star-States
    
radius = MSS[RADIUS_UNIT_INDEX,:,:] #Radius
#Extracting variables to be plotted over all 3 initial conditions and all mass steps.
variables = [
MSS[DENSITY_UNIT_INDEX,:,:], #Density
MSS[TEMP_UNIT_INDEX,:,:], #Temperature
MSS[PRESSURE_UNIT_INDEX,:,:], #Pressure
MSS[LUMINOSITY_UNIT_INDEX,:,:], #Luminosity
            ]
labels = ['Density', 'Temperature', 'Pressure', 'Luminosity']
units = ['g/cm³', 'K', 'Pa', 'W'] #Replace with actual units
#Plot dimensionless dependent variable vs. radius:
plt.figure(figsize=(20, 10)) 
plt.title('Stellar Radial Dependency of Density')
plt.xlabel('Radius') 
plt.ylabel('Density')
plt.grid(True)
for i in range(1,3):
    plt.plot(radius[:, i], variables[:, i], label=f'Initial Condition {i}') #Each variable is a 1xNx3 array, i.e. a 2D Nx3 array, so we iterate over each initial condition.
plt.legend()
plt.savefig('Density_R.png')


#multiply array by x_out array from Utilities for last question of problem.
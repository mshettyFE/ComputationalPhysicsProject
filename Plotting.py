import numpy as np
from matplotlib import pyplot as plt
import Integrator
from Utilities import *

if __name__ == "__main__":  # Main guard ensures this code runs only when the script is executed directly

    # Generate multi-star states with 3 sets of initial conditions. INPUT INITIAL CONDITIONS.
    state0 = Integrator.ODESolver()  
    state1 = Integrator.ODESolver()  
    state2 = Integrator.ODESolver()  

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


    plt.figure(figsize=(20, 10)) 
    for j, (variable, label, unit) in enumerate(zip(variables, labels, units)): #Index associated with each variable = j.
        plt.subplot(3, 2, j + 1) #Creates a 3x2 grid of subplots.
        for i in range(3): #Loop over the three initial conditions.
            plt.plot(radius[:, i], variable[:, i], label=f'Initial Condition {i+1}') #Each variable is a 1xNx3 array, i.e. a 2D Nx3 array, so we iterate over each initial condition.
        plt.xlabel('Radius (UNITS)') #Replace with actual units.
        plt.ylabel(f'{label} ({unit})')
        plt.title(f'Stellar Radial Dependency of {label}')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig('Radial_Dependency.png')
    plt.show()


#unloop this: one variable for each figure.
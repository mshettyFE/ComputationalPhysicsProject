import numpy as np
from matplotlib import pyplot as plt
import Integrator
import Minimizer
from Utilities import *

def plot_variable(independent, dependent, title, filename, xlabel, ylabel, logx = False, logy = False, clear= True):
    """
    Helper function to plot variables in the mesh against each other
    """
    if (clear):
        plt.clf()
    plt.figure(figsize=(20, 10)) 
    plt.title(title)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
#    plt.grid(True)
    plt.plot(independent, dependent)
    plt.savefig(filename+'.png')

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
conversion = UnitScalingFactors(M_sun, R_sun)
constants = generate_extra_parameters(M_sun, R_sun, E_0_sun, kappa_0_sun, mu_sun)
print(constants)
init_conds = Minimizer.gen_initial_conditions(1.5E7/conversion[TEMP_UNIT_INDEX], 26.5E6*1E9/conversion[PRESSURE_UNIT_INDEX],1E-2, constants)
print(init_conds)
state0 = Integrator.ODESolver(init_conds, 100, constants)
    
radius = state0[RADIUS_UNIT_INDEX,:] #Radius
#Extracting variables to be plotted over all 3 initial conditions and all mass steps.
variables = [
state0[DENSITY_UNIT_INDEX,:], #Density
state0[TEMP_UNIT_INDEX,:], #Temperature
state0[PRESSURE_UNIT_INDEX,:], #Pressure
state0[LUMINOSITY_UNIT_INDEX,:], #Luminosity
            ]
labels = ['Density', 'Temperature', 'Pressure', 'Luminosity']
units = ['g/cm³', 'K', 'Pa', 'W'] #Replace with actual units
plot_variable(radius, variables[0],'Stellar Radial Dependency of Density',"Density_R", 'Radius', 'Density', logy =True )
plot_variable(radius, variables[1],'Stellar Radial Dependency of Temperature',"Temp_R", 'Radius', 'Temp', logy =True )
plot_variable(radius, variables[2],'Stellar Radial Dependency of Pressure',"Pres_R", 'Radius', 'Pressure', logy =True )
plot_variable(radius, variables[3],'Stellar Radial Dependency of Luminosity',"Lum_R", 'Radius', 'Luminosity', logy =True )

print(variables[3])
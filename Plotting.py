import numpy as np
from matplotlib import pyplot as plt
from Utilities import *

def plot_variable(independent, dependent, title, filename, xlabel, ylabel, logx = False, logy = False, clear= True):
    """
    Helper function to plot variables in the mesh against each other
        Input:
            independent: 1D numpy array of the independent variable
            dependent: 1D numpy array of the dependent variable
            title: title of the plot (string)
            filename: Output file name of plot. Will write `filename`.png to current working directory
            xlabel: label of xaxis (string)
            ylabel: label of yaxis (string)
            logx, logx: flags to indicate if you want the x/y axes to be log
            clear: clear the previous plot before starting the new one. Set to False if you want to draw multiple curves with the same independent variable
        Output:
            .png file with filename in current directory
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
    plt.scatter(independent, dependent)
    plt.savefig(filename+'.png')

if __name__ == "__main__":  # Main guard ensures this code runs only when the script is executed directly
    state0 = np.loadtxt("SunMesh.txt", delimiter=",") # CHANGE THIS TO WHAT YOU NEED!
    radius = state0[:,RADIUS_UNIT_INDEX] # state0 is a Nx6 2D numpy array
    #Extracting variables to be plotted over all 3 initial conditions and all mass steps.
    variables = [
    state0[:,DENSITY_UNIT_INDEX], #Density
    state0[:,TEMP_UNIT_INDEX], #Temperature
    state0[:,PRESSURE_UNIT_INDEX], #Pressure
    state0[:,LUMINOSITY_UNIT_INDEX], #Luminosity
                ]
    labels = ['Density', 'Temperature', 'Pressure', 'Luminosity']
    units = ['g/cm³', 'K', 'Pa', 'W'] #Replace with actual units
    plot_variable(radius, variables[0],'Stellar Radial Dependency of Density',"Density_R", 'Radius', 'Density', logy =True, logx=True )
    plot_variable(radius, variables[1],'Stellar Radial Dependency of Temperature',"Temp_R", 'Radius', 'Temp', logy =True , logx=True)
    plot_variable(radius, variables[2],'Stellar Radial Dependency of Pressure',"Pres_R", 'Radius', 'Pressure', logy =True , logx=True)
    plot_variable(radius, variables[3],'Stellar Radial Dependency of Luminosity',"Lum_R", 'Radius', 'Luminosity', logy =True , logx=True)
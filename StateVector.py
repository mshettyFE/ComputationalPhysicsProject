import numpy as np
import sys
from enum import Enum, auto
from Utilities import equation_of_state

class StateVectorVar(Enum):
    RADIUS= 0
    PRESSURE=1
    TEMP=2
    LUMINOSITY=3

class InterpolationIndex(Enum):
    RADIUS=0
    PRESSURE=1
    TEMP=2
    LUMINOSITY=3
    DENSITY=4



class StateVector():
    def __init__(self, n_shells, test_data=False):
#        Assumes that state_vector has the following form:
#        <r_0, r_1,...r_{k-1}, P_0,P_1,...P_{k-1}, T_0,...T_{k-1}, L_0,...L_{k-1}>, where k is the number of shells
        self.n_shells = n_shells
        if(test_data):
            rad = 2*np.arange(n_shells)
            pres = 3*np.arange(n_shells)
            temp = 5*np.arange(n_shells)
            lum = 7*np.arange(n_shells)
            output = np.concatenate([rad, pres, temp, lum], axis=None)
        else:
            #TODO: Makes these guesses more realistic. For now, just panic
            print("Make better guesses please")
            sys.exit(1)
        starting_indices = self.gen_starting_index()
        self.state_vec, self.starting_indices = output, starting_indices

    def gen_starting_index(self):
        """
            Generate starting indices into state_vector for each variable (r,P,T,L)
            Output:
                starting_indices: a map from StateVectorVar to the starting index in state_vector
        """
        starting_indices = {}
        starting_indices[StateVectorVar.RADIUS] = 0
        starting_indices[StateVectorVar.PRESSURE] = self.n_shells
        starting_indices[StateVectorVar.TEMP] = self.n_shells*2
        starting_indices[StateVectorVar.LUMINOSITY] = self.n_shells*3
        return starting_indices
    
    def update_state(self, vector):
        assert(vector.shape==self.state_vec.shape)
        self.state_vec = vector

    def stitch_vector(self):
        """
            Removing r_0,P_{k-1}, T_{k-1},L_0 where k is the number of shells.
            This is done to preserve the boundary conditions of the system (the removed elements are all 0).
            By removing these elements, we prevent the Jacobian from going singular.
            Output:
                out_vec: 4*(k-1) dimensional array
        """
        dim = self.state_vec.shape[0]
        assert(dim != 0)
        assert(dim%4==0)
        n_shells = int(dim/4)
        reduced_sv = np.zeros(4*(n_shells-1))

        starting_rad = self.starting_indices[StateVectorVar.RADIUS]
        starting_p = self.starting_indices[StateVectorVar.PRESSURE]
        starting_temp = self.starting_indices[StateVectorVar.TEMP]
        starting_lum = self.starting_indices[StateVectorVar.LUMINOSITY]

        rad_part = self.state_vec[starting_rad+1:starting_rad+n_shells] # exclude r_0
        pressure_part = self.state_vec[starting_p:starting_p+n_shells-1] # exclude P_{k-1}
        temp_part = self.state_vec[starting_temp:starting_temp+n_shells-1] # exclude T_{k-1}
        lum_part = self.state_vec[starting_lum+1:starting_lum+n_shells] # exclude L_0

        # Need to subtract off from the starting index as you go up to have proper indexing
        reduced_sv[starting_rad:starting_rad+n_shells-1] = rad_part
        reduced_sv[starting_p-1:starting_p+n_shells-2] = pressure_part 
        reduced_sv[starting_temp-2:starting_temp+n_shells-3] = temp_part
        reduced_sv[starting_lum-3:starting_lum+n_shells-4] = lum_part
        return reduced_sv

    def unstitch_vector(self,reduced_state_vector):
        """
            Given a reduced state vector, insert 0's in the appropriate places so that the dimension matches of self.state_vec
            Input:
                reduced_state_vector: 4(k-1) dimensional array where k is the number of mass shells
                starting_indices: The starting index of each variable for the state vector, NOT the reduced_state_vector (use gen_starting_index to produce)
            Output:
                out_vec: 4k dimensional array, with 0's in appropriate places
        """
        dim = reduced_state_vector.shape[0]
        assert(dim != 0)
        assert(dim%4==0)
        n_shells_minus_one = int(dim/4)
        n_shells = n_shells_minus_one+1
        output = np.zeros(4*n_shells)


        starting_rad = self.starting_indices[StateVectorVar.RADIUS]
        starting_p = self.starting_indices[StateVectorVar.PRESSURE]
        starting_temp = self.starting_indices[StateVectorVar.TEMP]
        starting_lum = self.starting_indices[StateVectorVar.LUMINOSITY]

        # Include additional offset to account for the fact that starting_indices is for original index
        starting_rad_reduced = starting_rad
        starting_p_reduced = starting_p-1
        starting_temp_reduced = starting_temp-2
        starting_lum_reduced = starting_lum-3

        rad_part = reduced_state_vector[starting_rad_reduced: starting_rad_reduced+n_shells_minus_one]
        p_part = reduced_state_vector[starting_p_reduced: starting_p_reduced+n_shells_minus_one]
        t_part = reduced_state_vector[starting_temp_reduced: starting_temp_reduced+n_shells_minus_one]
        lum_part = reduced_state_vector[starting_lum_reduced: starting_lum_reduced+n_shells_minus_one]

        output[starting_rad+1: starting_rad+1+n_shells_minus_one] = rad_part # Don't touch r0
        output[starting_p: starting_p+n_shells_minus_one] = p_part # Don't touch p_{k-1}
        output[starting_temp: starting_temp+n_shells_minus_one] = t_part # Don't touch T_{k-1}
        output[starting_lum+1: starting_lum+1+n_shells_minus_one] = lum_part# Don't touch r0

        return output

    def interpolate(self, which_var):
        """
            generate average of adjacent elements in a vector
            Input:
                which_var: StateVectorVar
            Output:
                n-1 dimensional np array of interpolated variable
        """
        first = self.state_vec[self.starting_indices[which_var]:self.starting_indices[which_var]+self.n_shells-1]
        second = self.state_vec[self.starting_indices[which_var]+1:self.starting_indices[which_var]+self.n_shells]
        return (first+second)/2
    
    def interpolate_all(self, constants):
        """
            generate the interpolation for all variables, including density
            Inputs:
                constants: dictionary produced by Utilities.UnitScalingFactors
            Output:
                5x(k-1) dimensional array, where each column is the interpolation of that particular variable.
                Indexed by InterpolationIndex
        """
        rad = self.interpolate(StateVectorVar.RADIUS)
        pres = self.interpolate(StateVectorVar.PRESSURE)
        temp = self.interpolate(StateVectorVar.TEMP)
        lum = self.interpolate(StateVectorVar.LUMINOSITY)
        density = equation_of_state(pres, temp, constants)
        return np.vstack([rad, pres, temp, lum, density])
        
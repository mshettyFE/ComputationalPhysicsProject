import jax.numpy as jnp
import numpy as np
import sys
from enum import Enum
from Utilities import equation_of_state

# Allows you to index self.state_vec to extract the associated variables
class StateVectorVar(Enum):
    RADIUS= 0
    PRESSURE=1
    TEMP=2
    LUMINOSITY=3

# Allows you to index into the interpolated values returned from StateVector.interpolate_all()
class InterpolationIndex(Enum):
    RADIUS= 0
    PRESSURE=1
    TEMP=2
    LUMINOSITY=3
    DENSITY=4
    MASS=5

class DataGenMode(Enum):
    TEST=0
    RANDOM=1
    LINEAR=2
    LOG=3   
    PHYSICAL=4

class StateVector():
    def __init__(self, n_shells, data_gen_type:DataGenMode):
        """
            Members:
                state_vec: a 4*n_shells dimensional vector containing the values of each variable at each shell
                    Assumes that state_vector has the following form:
                    <r_0, r_1,...r_{k-1}, P_0,P_1,...P_{k-1}, T_0,...T_{k-1}, L_0,...L_{k-1}>, where k is the number of shells
                n_shells: the number of shells used
                shell_masses: the mass value assigned to each shell
                starting_indicies: A dictionary housing at what index does each variable begin in the state vector
        """
        self.n_shells = n_shells
        self.dm = 1/self.n_shells
        self.data_gen_mode = data_gen_type
        if(data_gen_type==DataGenMode.TEST):
            # Dummy data to see of can run end to end
            output = jnp.zeros((4*n_shells))
            output =output.at[0:n_shells].add(2*jnp.arange(n_shells))
            output =output.at[n_shells:2*n_shells].add(3*jnp.arange(n_shells))
            output =output.at[2*n_shells:3*n_shells].add(5*jnp.arange(n_shells))
            output =output.at[3*n_shells:4*n_shells].add(7*jnp.arange(n_shells))
        elif (data_gen_type==DataGenMode.RANDOM):
            # Fill all variables with random uniform noise.
            output = jnp.zeros((4*n_shells))
            output = output.at[0:n_shells].add(jnp.array(np.random.rand(n_shells)))
            output = output.at[n_shells:2*n_shells].add(jnp.array(np.random.rand(n_shells)))
            output = output.at[2*n_shells:3*n_shells].add(jnp.array(np.random.rand(n_shells)))
            output = output.at[3*n_shells:4*n_shells].add(jnp.array(np.random.rand(n_shells)))
            # Apply boundary conditions to state vector
            output = output.at[0].set(0) # r_0 = 0
            output = output.at[2*n_shells-1].set(0) # P_k-2 = 0
            output = output.at[3*n_shells-1].set(0) # T_k-2 = 0
            output = output.at[3*n_shells].set(0) # L_0 = 0
        elif (data_gen_type==DataGenMode.LINEAR):
            # Assumes that each variable follows a linear slope, increasing or decreasing as appropriate
            output = jnp.zeros((4*n_shells))
            linear = jnp.linspace(0,1,n_shells)
            rev_linear = jnp.flip(linear)
            output = output.at[0:n_shells].add(linear)
            output = output.at[n_shells:2*n_shells].add(rev_linear)
            output = output.at[2*n_shells:3*n_shells].add(rev_linear)
            output = output.at[3*n_shells:4*n_shells].add(linear)
            # Apply boundary conditions to state vector
            output = output.at[0].set(0) # r_0 = 0
            output = output.at[2*n_shells-1].set(0) # P_k-2 = 0
            output = output.at[3*n_shells-1].set(0) # T_k-2 = 0
            output = output.at[3*n_shells].set(0) # L_0 = 0
        elif (data_gen_type==DataGenMode.LOG):
            # Assumes that each variable follows a linear slope, increasing or decreasing as appropriate
            # Added twist is to cast everything to log variables
            output = jnp.zeros((4*n_shells))
            linear = jnp.linspace(1E-9,1,n_shells) # lower bound to take into account that log(0) is undefined
            rev_linear = jnp.flip(linear)
            log = jnp.log(linear)
            rev_log = jnp.log(rev_linear)
            output = output.at[0:n_shells].add(log)
            output = output.at[n_shells:2*n_shells].add(rev_log)
            output = output.at[2*n_shells:3*n_shells].add(rev_log)
            output = output.at[3*n_shells:4*n_shells].add(log)
        elif (data_gen_type==DataGenMode.PHYSICAL):
            # Have pressure fall off faster than temperature
            output = jnp.zeros((4*n_shells))
            linear = jnp.linspace(0,1,n_shells)
            exp = jnp.exp(-2*linear)
            exp_fast = jnp.exp(-4*linear)
            output = output.at[0:n_shells].add(linear)
            output = output.at[n_shells:2*n_shells].add(exp_fast)
            output = output.at[2*n_shells:3*n_shells].add(exp) 
            output = output.at[3*n_shells:4*n_shells].add(linear)
            output = output.at[0].set(1E-9) # r_0 = 0
            output = output.at[2*n_shells-1].set(1E-9) # P_k-2 = 0
            output = output.at[3*n_shells-1].set(1E-9) # T_k-2 = 0
            output = output.at[3*n_shells].set(1E-9) # L_0 = 0
        else:
            print("Undefined data generation mode")
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
        
    def update_state(self, delta, check_neg = True):
        """
            Try and update state vector with another delta. Check if any values go negative
            delta: 1D array that's the same size as self.state_vec
            check_neg: flag to run negativity check
        """
        assert(delta.shape==self.state_vec.shape)
        temp = self.state_vec+delta
        if(jnp.any(temp<0) and check_neg):
            return 0
        self.state_vec = temp
        return 1



    def stitch_vector(self,vector):
        """
            Removing r_0,P_{k-1}, T_{k-1},L_0 where k is the number of shells.
            This is done to preserve the boundary conditions of the system (the removed elements are all 0).
            By removing these elements, we prevent the Jacobian from going singular.
            Input:
                vector: 4*k dimensional array
            Output:
                out_vec: 4*(k-1) dimensional array
        """
        dim = vector.shape[0]
        assert(dim == self.state_vec.shape[0])
        reduced_sv = jnp.zeros(4*(self.n_shells-1))
        starting_rad = self.starting_indices[StateVectorVar.RADIUS]
        starting_p = self.starting_indices[StateVectorVar.PRESSURE]
        starting_temp = self.starting_indices[StateVectorVar.TEMP]
        starting_lum = self.starting_indices[StateVectorVar.LUMINOSITY]
        rad_part = vector[starting_rad+1:starting_rad+self.n_shells] # exclude r_0
        pressure_part = vector[starting_p:starting_p+self.n_shells-1] # exclude P_{k-1}
        temp_part = vector[starting_temp:starting_temp+self.n_shells-1] # exclude T_{k-1}
        lum_part = vector[starting_lum+1:starting_lum+self.n_shells] # exclude L_0
        # Need to subtract off from the starting index as you go up to have proper indexing
        reduced_sv = reduced_sv.at[starting_rad:starting_rad+self.n_shells-1].add(rad_part)
        reduced_sv = reduced_sv.at[starting_p-1:starting_p+self.n_shells-2].add(pressure_part)
        reduced_sv = reduced_sv.at[starting_temp-2:starting_temp+self.n_shells-3].add(temp_part)
        reduced_sv = reduced_sv.at[starting_lum-3:starting_lum+self.n_shells-4].add(lum_part)
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
        assert(dim+4==self.state_vec.shape[0])
        n_shells_minus_one = self.n_shells-1
        output = jnp.zeros(4*self.n_shells)
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
        output = output.at[starting_rad+1: starting_rad+self.n_shells].add(rad_part) # Don't touch r0
        output = output.at[starting_p: starting_p+self.n_shells-1].add(p_part) # Don't touch p_{k-1}
        output = output.at[starting_temp: starting_temp+self.n_shells-1].add(t_part) # Don't touch T_{k-1}
        output = output.at[starting_lum+1: starting_lum+self.n_shells].add(lum_part) # Don't touch r0
        return output
    
    def stich_jacobian(self, jac_matrix):
        """
            Remove associated rows and columns of jacobian matrix to make it compatible with boundary conditions
            Input:
                jac_matrix: 4k*4k dimensional matrix
                output: 4*(k-1) square matrix
        """
        starting_rad = self.starting_indices[StateVectorVar.RADIUS]
        starting_p = self.starting_indices[StateVectorVar.PRESSURE]
        starting_temp = self.starting_indices[StateVectorVar.TEMP]
        starting_lum = self.starting_indices[StateVectorVar.LUMINOSITY]
        indicies = jnp.array([starting_rad, starting_p+self.n_shells-1,starting_temp+self.n_shells-1, starting_lum])
        del_rows = jnp.delete(jac_matrix, indicies , axis=0)
        output =  jnp.delete(del_rows, indicies, axis=1)
        return output

    def extract_variable(self, which_var):
        """
            grab the specified variable array
            Input:
                which_var: StateVectorVar
            Output:
                n_shells dimensional np array of variable
        """
        return self.state_vec.at[self.starting_indices[which_var]:self.starting_indices[which_var]+self.n_shells]

    def save_state(self, filename):
        """
            Wrapper around jnp.savetxt()
        """
        jnp.save(filename, self.state_vec)

    def load_state(self, filename):
        """
            Wrapper around jnp.loadtxt()
        """
        cand_state = jnp.load(filename)
        assert (cand_state.shape[0]%4 ==0)
        self.n_shells = int(cand_state.shape[0]/4)
        self.state_vec = cand_state
        self.starting_indices = self.gen_starting_index()
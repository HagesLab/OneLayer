# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:03:34 2021

@author: cfai2
"""
import numpy as np
from scipy import integrate as intg
from _helper_structs import Parameter, Output
from utils import u_read, to_index, to_array, to_pos
import tables
from _OneD_Model import OneD_Model

class HeatPlate(OneD_Model):
    
    def __init__(self):
        super().__init__()
        self.system_ID = "HeatPlate (Const Bound Flux)"
        self.length_unit = "[m]"
        self.time_unit = "[s]"
        
        self.param_dict = {"k":Parameter(units="[W / m k]", is_edge=False), "Cp":Parameter(units="[J / kg K]", is_edge=False), 
                            "density":Parameter(units="[kg m^-3]", is_edge=False), "init_T":Parameter(units="[K]", is_edge=False),
                            "Left_flux":Parameter(units="[W m^-2]", is_edge=False), "Right_flux":Parameter(units="[W m^-2]", is_edge=False)}
        
        self.param_count = len(self.param_dict)
        
        self.flags_dict = {"symmetric_system":("Symmetric System",1, 0)}

        # List of all variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        self.simulation_outputs_dict = {"T":Output("Temperature", units="[K]", xlabel="m", xvar="position", is_edge=False, yscale='linear')}
        
        # List of all variables calculated from those in simulation_outputs_dict
        self.calculated_outputs_dict = {"q":Output("Heat Flux", units="[W/m^2]", xlabel="m", xvar="position", is_edge=True)}
        
        self.outputs_dict = {**self.simulation_outputs_dict, **self.calculated_outputs_dict}
        
        self.simulation_outputs_count = len(self.simulation_outputs_dict)
        self.calculated_outputs_count = len(self.calculated_outputs_dict)
        self.total_outputs_count = self.simulation_outputs_count + self.calculated_outputs_count
        ## Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        self.convert_in_dict = {"k":1, "Cp":1, "density":1, "init_T":1, "T":1, "q":1, "Left_flux":1, "Right_flux":1}

        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
        self.convert_out_dict = {}
        for param in self.convert_in_dict:
            self.convert_out_dict[param] = self.convert_in_dict[param] ** -1

        return
    
    def calc_inits(self):
        """ Package initial temperature distribution"""
        init_T = self.param_dict['init_T'].value * self.convert_in_dict["T"]
        init_T = to_array(init_T, len(self.grid_x_nodes), False)
        return {"T":init_T}
    
    def simulate(self, data_path, m, n, dt, params, flags, hmax_, init_conditions):
        # No strict rules on how simulate() needs to look - as long as it calls the appropriate ode() from py with the correct args
        return ode_heatplate(data_path, m, n, self.dx, dt, params)
    
    def get_overview_analysis(self, params, tsteps, data_dirname, file_name_base):
        """Calculate temperature and heat flux at various sample times"""
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        data_dict = {}
        
        for raw_output_name in self.simulation_outputs_dict:
            data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, 
                                                 raw_output_name)
            data = []
            for tstep in tsteps:
                data.append(u_read(data_filename, t0=tstep, single_tstep=True))
            
            data_dict[raw_output_name] = np.array(data)
            

        data_dict["Heat FLux"] = heatflux(data_dict, params)
                
                
        for data in data_dict:
            data_dict[data] *= self.convert_out_dict[data]
            
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, for_integrate=False, i=0, j=0,
                     nen=False, extra_data = None):
        """ Calculate heat flux (or read in temperature) on demand"""
        if (datatype in self.simulation_outputs_dict):
            data = sim_data[datatype]
        
        else:
            if (datatype == "q"):
                data = heatflux(sim_data, params)

            else:
                raise ValueError
                
        return data
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        """ Set temperature distribution of outgoing IC file"""
        param_dict["T"] = sim_data["T"] if include_flags['T'] else np.zeros(len(grid_x))

        return
    
def heat_constflux(t, y, m, dx, k, rho, Cp, q0, qL):
    """ derivative function for heat diffusion model"""
    alpha = k * rho / Cp
    G = alpha / (dx**2)
    T = y
    
    dydt = np.zeros(m)
    for i in range(1, len(T) - 1):
        dydt[i] = G[i+1]*T[i+1] - 2*G[i]*T[i] + G[i-1]*T[i-1]
    
    # Bounds
    dydt[0] = dydt[1]
    dydt[-1] = dydt[-2]
    
    return dydt
    
def ode_heatplate(data_path_name, m, n, dx, dt, params, write_output=True):
    """
    Master function for Neumann boundary heat problem module simulation.
    Problem statement:
    Create a discretized, time and space dependent solution (T(x,t) and P(x,t)) of a one-dimensional heated object with m space steps and n time steps
    Space step size is dx, time step is dt
    Initial conditions: init_T

    Parameters
    ----------
    data_path_name : str
        Output file location.
    m : int
        Number of space nodes.
    n : int
        Number of time steps.
    dx : float
        Space node width.
    dt : float
        Time step size.
    params : dict {"str":float or 1D array}
        Collection of parameter values
    write_output : bool, optional
        Whether to write output files. TEDs always does this but other applications reusing this function might not. The default is True.


    Returns
    -------
    None
        TEDs does not do anything with the return value. Other applications might find this useful however.

    """
    atom = tables.Float64Atom()

    ## Unpack params; typecast non-array params to arrays if needed
    q0 = params["Left_flux"]
    qL = params["Right_flux"]
    init_T = to_array(params["init_T"], m, False)

    k = to_array(params["k"], m, False)
    Cp = to_array(params["Cp"], m, False)
    rho = to_array(params["density"], m, False)
    
    init_T[0] = init_T[1] + q0*dx/k[0]
    init_T[-1] = init_T[-2] - qL*dx/k[-1]
    
    tSteps = np.linspace(0, n*dt, n+1)
    data, error_data = intg.odeint(heat_constflux, init_T, tSteps, 
                                   args=(m, dx, k, rho, Cp, q0, qL),
                                   tfirst=True, full_output=True)
            
    if write_output:
        ## Prep output files
        with tables.open_file(data_path_name + "-T.h5", mode='a') as ofstream_T:
            array_T = ofstream_T.root.data

            array_T.append(data[1:])

        return error_data

    else:
        array_T = data

        return array_T, error_data
    return

def heatflux(sim_data, params):
    """Calculate heat flux from temperature distribution using Newton's law"""
    T = sim_data['T']
    k = to_array(params['k'], len(T), False)
    k_avg = np.zeros(len(T) - 1)
    for i in range(0, len(T) - 1):
        k_avg[i] = (k[i] + k[i+1]) / 2
        
    if T.ndim == 1:
        q = np.zeros(len(T) + 1)
        q[0] = params["Left_flux"]
        q[-1] = params["Right_flux"]
        for i in range(1, len(q) - 1):
            q[i] = -k_avg[i-1] * (T[i] - T[i-1]) / params["Node_width"]
            
    elif T.ndim == 2:
        q = np.zeros((len(T), len(T[0]) + 1))
        q[:,0] += params["Left_flux"]
        q[:,-1] += params["Right_flux"]
        for i in range(1, len(q[0]) - 1):
            q[:,i] = -k_avg[i-1] * (T[:,i] - T[:,i-1]) / params["Node_width"]
        
    return q
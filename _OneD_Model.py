# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:25 2021

@author: cfai2
"""
import numpy as np
from helper_structs import Parameter, Output
from utils import u_read, to_index, to_array, to_pos
import tables

class OneD_Model:
    """ Template class for modules.
    
    Modules act as blueprints used to create simulated models.
    
    This class stores information regarding a module's 
    Layers, Parameters, Outputs, and Flags.
    
    Also stores information used to build the initial state of a model system,
    such as space grids and parameter rules/distributions.
    
    All child modules must populate __init__'s dicts and implement:
        calc_inits()
        simulate()
        get_overview_analysis()
        prep_dataset()
        get_IC_carry()
    """
    
    def __init__(self):
        # Unique identifier for module.
        self.system_ID = "INSERT MODEL NAME HERE"  
        
        self.time_unit = "[ns]"
        self.layers = {}
        # Parameter list.
        params = {}

        # dict {"Flag's internal name":
        #       ("Flag's display name",whether Flag1 is toggleable, Flag1's default value)
        #      }
        self.flags_dict = {"Flag1":("Flag1's name",1, 0)}
        
        # dict {"Variable name":Output()} of all dependent variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        simulation_outputs = {}
        
        # dict {"Variable name":Output()} of all secondary variables calculated from those in simulation_outputs_dict
        calculated_outputs = {}

        ## Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        # Each item in param_dict and outputs_dict must have an entry here, even if the conversion factor is one
        convert_in = {}
        
        # The integration tool uses whatever units self.dx is in.
        # Define the "integration_scale" entry to correct for any mismatches
        # between the integrand's and self.dx's length units.
        convert_in["integration_scale"] = 1

        return
    
    def add_param_rule(self, layer, param_name, new_rule):
        """

        Parameters
        ----------
        layer : str
            Layer from which param should be selected.
        param_name : str
            Name of a parameter from a layers' params dict.
        new_rule : helper_structs.Parameter()
            New Parameter() instance created from TEDs.add_paramrule()

        Returns
        -------
        None.

        """
        self.layers[layer].params[param_name].param_rules.append(new_rule)
        self.update_param_toarray(layer, param_name)
        return

    def swap_param_rules(self, layer, param_name, i):
        """

        Parameters
        ----------
        layer : str
            Layer from which param should be selected.
        param_name : str
            Name of a parameter from a layer's param dict
        i : int
            Index of parameter rule to be swapped with i-1

        Returns
        -------
        None.

        """
        param = self.layers[layer].params[param_name]
        param.param_rules[i], param.param_rules[i-1] = param.param_rules[i-1], param.param_rules[i]
        self.update_param_toarray(layer, param_name)
        return

    def remove_param_rule(self, layer, param_name, i):
        """

        Parameters
        ----------
        layer : str
            Layer from which param should be selected.
        param_name : str
            Name of a parameter from a layer's param dict
        i : int
            Index of parameter rule to be deleted

        Returns
        -------
        None.

        """
        self.layers[layer].params[param_name].param_rules.pop(i)
        self.update_param_toarray(layer, param_name)
        return
    
    def removeall_param_rules(self, layer, param_name):
        """
        Deletes all parameter rules and resets stored distribution for a given parameter

        Parameters
        ----------
        layer : str
            Layer from which param should be selected.
        param_name : str
            Name of a parameter from a layer's param dict

        Returns
        -------
        None.

        """
        self.layers[layer].params[param_name].param_rules = []
        self.layers[layer].params[param_name].value = 0
        return
    
    def update_param_toarray(self, layer, param_name):
        """ Recalculate a Parameter from its Param_Rules"""
        # This should be done every time the Param_Rules are changed
        # All params calculated in this fashion are stored as array, 
        # even if the param is space invariant
        param = self.layers[layer].params[param_name]

        if param.is_edge:
            new_param_value = np.zeros(len(self.layers[layer].grid_x_edges))
        else:
            new_param_value = np.zeros(len(self.layers[layer].grid_x_nodes))

        for condition in param.param_rules:
            i = to_index(condition.l_bound, self.layers[layer].dx, 
                         self.layers[layer].total_length, param.is_edge)
            
            # If the left bound coordinate exceeds the width of the node toIndex() (which always rounds down) 
            # assigned, it should actually be mapped to the next node
            if (condition.l_bound - to_pos(i, self.layers[layer].dx, param.is_edge) > self.layers[layer].dx / 2): 
                i += 1

            if (condition.type == "POINT"):
                new_param_value[i] = condition.l_boundval

            elif (condition.type == "FILL"):
                j = to_index(condition.r_bound, self.layers[layer].dx, self.layers[layer].total_length, param.is_edge)
                new_param_value[i:j+1] = condition.l_boundval

            elif (condition.type == "LINE"):
                slope = (condition.r_boundval - condition.l_boundval) / (condition.r_bound - condition.l_bound)
                j = to_index(condition.r_bound, self.layers[layer].dx, self.layers[layer].total_length, param.is_edge)

                ndx = np.linspace(0, self.layers[layer].dx * (j - i), j - i + 1)
                new_param_value[i:j+1] = condition.l_boundval + ndx * slope

            elif (condition.type == "EXP"):
                j = to_index(condition.r_bound, self.layers[layer].dx, self.layers[layer].total_length, param.is_edge)

                ndx = np.linspace(0, j - i, j - i + 1)
                try:
                    new_param_value[i:j+1] = condition.l_boundval * np.power(condition.r_boundval / condition.l_boundval, ndx / (j - i))
                except FloatingPointError:
                    print("Warning: Step size too large to resolve initial condition accurately")

        param.value = new_param_value
        return
    
    def count_s_outputs(self):
        count = 0
        for layer_name, layer in self.layers.items():
            count += layer.s_outputs_count
        return count
    
    def find_layer(self, output):
        for layer_name, layer in self.layers.items():
            if output in layer.outputs:
                return layer_name
    def DEBUG_print(self):
        """
        Prepares a summary of the current state of all parameters.
        
        For TEDs state summary button/popup.

        Returns
        -------
        mssg : str
            Message displayed in popup.

        """
        
        
        mssg = []
        for layer in self.layers:
            mssg.append("LAYER: {}".format(layer))
            if self.layers[layer].spacegrid_is_set:
                mssg.append("Space Grid is set")
                if len(self.layers[layer].grid_x_nodes) > 20:
                    mssg.append("Nodes: [{}...{}]".format(", ".join(map(str,self.layers[layer].grid_x_nodes[0:3])),
                                                        ", ".join(map(str,self.layers[layer].grid_x_nodes[-3:]))))
                    mssg.append("Edges: [{}...{}]".format(", ".join(map(str,self.layers[layer].grid_x_edges[0:3])),
                                                        ", ".join(map(str,self.layers[layer].grid_x_edges[-3:]))))
                    
                else:
                    mssg.append("Nodes: {}".format(self.layers[layer].grid_x_nodes))
                    mssg.append("Edges: {}".format(self.layers[layer].grid_x_edges))
            else:
                mssg.append("Grid is not set")
    
            for param in self.layers[layer].params:
                mssg.append("{} {}: {}".format(param, self.layers[layer].params[param].units,
                                               self.layers[layer].params[param].value))
            mssg.append("#########")
        return "\n".join(mssg)
    
    def calc_inits(self):
        """
        Uses the self.param_dict to calculate initial conditions for ODEINT.
        
        In many cases this is just returning the appropriate parameter from self.param_dict.

        Returns
        -------
        dict {"param name":1D numpy array}
            Collection of initial condition arrays.

        """
        
        return {}
    
    def simulate(self, data_path, m, n, dt, params, flags, do_ss, hmax_, init_conditions):
        """
        Uses the provided args to call a numerical solver and write results to .h5 files.
        
        No strict rules on how simulate() needs to look or if all parameters must be utilised - 
        nor is ODEINT() required (although these are suggested) - 
        as long as this function simulates and writes output files.
        
        Parameters
        ----------
        data_path : str
            Absolute path to target directory files are written in.
        m : int
            Number of space grid nodes.
        n : int
            Number of time steps.
        dt : float
            Time step size.
        params : dict {"param name": 1D numpy array} or dict {"param name": float}
            Collection of parameter values.
        flags : dict {"Flag internal name": Flag()}
            Collection of flag instances.
        do_ss : bool
            Special steady state injection flag.
        hmax_ : float
            Maximum time stepsize to be taken by ODEINT().
        init_conditions : dict {"param name": 1D numpy array}
            Collection of initial conditions.

        Returns
        -------
        None.

        """

        return
    
    def get_overview_analysis(self, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
        """
        Perform and package all calculations to be displayed on TEDs OVerview Analysis tab.

        Parameters
        ----------
        params : dict {"param name": 1D numpy array or float}
            Paramteres used in simulation and stored in metadata.
        flags : dict {"flag name": int}
            Flags used in simulation and stored in metadata.
        total_time : float
            Total time of simulation
        dt : float
            Timestep used in simulation
        tsteps : list, 1D numpy array
            Time step indices overview should sample over
        data_dirname : str
            Name of output data directory e.g. .../Data/sim1.
        file_name_base : str
            Name of simulation file e.g. mydata.
            Used with data_dirname to locate file - files are written in format
            data_dirname/file_name_base-output_name.h5
            e.g. .../Data/mysimulation1-y.h5

        Returns
        -------
        data_dict : dict {"output name": 1D or 2D numpy array}
            Collection of output data. 1D array if calculated once (e.g. an integral over space) and 2D if calculated at multiple times
            First dimension is time, second dimension is space
            Must return one entry per output in self.output_dict
        """
        data_dict = {}

        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, flags, for_integrate=False, i=0, j=0, nen=False, extra_data = None):
        """
        Use the raw data in sim_data to calculate quantities in self.outputs_dict.
        If datatype is in self.simulated_outputs dict this just needs to return the correct item from sim_data

        Parameters
        ----------
        datatype : str
            The item we need to calculate.
        sim_data : dict {"datatype": 1D or 2D numpy array}
            Collection of raw data read from .h5 files. 1D if single time step or 2D if time and space range.
        params : dict {"param name": 1D numpy array}
            Collection of param values from metadata.txt.
        flags : dict {"flag name": int}
            Collection of flag values from metadata.txt.
        for_integrate : bool, optional
            Whether this function is being called by do_Integrate. Used for integration-specific procedures.
            All other optional arguments are needed only if for_integrate is True.
            The default is False.
        i : int, optional
            Left bound index for integral. The default is 0.
        j : int, optional
            Right bound index for integral. The default is 0.
        nen : bool, optional
            Short for "need extra node". Whether an extra (the 'j+1'th) node should be counted.
            This is determined by a correction between the node values and the actual bounds of the integration.
            See Nanowire.prep_PL and GUI's do_integrate for an example of this.
            The default is False.
        extra_data : dict {"datatype": 2D numpy array}, optional
            A copy of the full set of raw data over all time and space. Some integrals like the weighted PL need this.
            The default is None.

        Returns
        -------
        data : 1D or 2D numpy array
            Finalized data set. 2D if for_integrate and 1D otherwise.

        """
        data = None
        
        # data = (your calcs here)
                
        return data
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        """
        Overwrites param_dict with values from the current data analysis in preparation for generating
        a new initial state file.

        Parameters
        ----------
        sim_data : dict {"data type": 1D numpy array}
            Collection of raw data from the currently loaded timestep of the currently loaded dataset.
        param_dict : dict {"param name": 1D numpy array}
            Collection of params from the currently loaded dataset.
        include_flags : dict {"data type": bool}
            Whether to include a data type from self.simulated_outputs_dict in the new initial file.
            One per output in self.simulated_outputs_dict
        grid_x : 1D numpy array
            Space node grid for currently loaded dataset.

        Returns
        -------
        None.

        """
        param_dict["Output1"] = None
        return
    
    def verify(self):
        """Performs basic syntactical checks on module's attributes."""
        print("Verifying selected module...")
        
        
        if not "symmetric_system" in self.flags_dict:
            print("Warning: no symmetric_system flag defined."
                  "Automatically setting symmetric_system to FALSE")
            self.flags_dict["symmetric_system"] = ("Symmetric System", 0, 0)
        
        for layer in self.layers:
            if not "integration_scale" in self.layers[layer].convert_in:
                print("Warning: no integration_scale correction defined for layer {}. "
                      "Integration may have incorrect units.".format(layer))
                self.layers[layer].convert_in["integration_scale"] = 1
                self.layers[layer].convert_out["integration_scale"] = 1
        return
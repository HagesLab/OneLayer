# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:25 2021

@author: cfai2
"""
import numpy as np
from helper_structs import Parameter, Output, Layer
from utils import to_index, to_pos

from config import init_logging
logger = init_logging(__name__)

class OneD_Model:
    """ Template class for modules.
    
    Modules act as blueprints used to create simulated models.
    
    This class stores information regarding a module's 
    Layers, Parameters, Outputs, and Flags.
    
    All child modules must, in  __init__, define:
        system_ID
        time_unit
        flags_dict
        layers
        shared_layer
        
    and implement:
        calc_inits()
        simulate()
        get_overview_analysis()
        prep_dataset()
        get_timeseries()
        get_IC_carry()
    """
    
    def __init__(self):
        # Unique identifier for module.
        self.system_ID = "INSERT MODEL NAME HERE"  
        
        # Internal time unit for this module's solvers.
        
        self.time_unit = "[ns]"
        
        # List of layers in this module.
        self.layers = {}

        params = {"Example Parameter": Parameter(units="[cm / s]", is_edge=False, valid_range=(0,np.inf))}

        # dict {"Flag's internal name":
        #       ("Flag's display name",whether Flag1 is toggleable, Flag1's default value)
        #      }
        self.flags_dict = {"Flag1":("Flag1's name",1, 0)}
        

        simulation_outputs = {"Example Output":Output("Output name", units="[cm^-3]", integrated_units="[cm^-2]",
                                                      xlabel="cm", xvar="position", is_edge=False, 
                                                      layer="Example Layer", yscale='symlog', yfactors=(1e-4,1e1))}
        calculated_outputs = {}
        convert_in = {"Example Parameter": 1e-2, # [cm / s] to [nm / ns]
                      "Example Output": 1e-21, # [cm^-3] to [nm^-3]
                      }
        iconvert_in = {"Example Output": 1e7 # [cm] to [nm]
                       }
        
        self.layers["Example Layer"] = Layer(params, simulation_outputs, calculated_outputs, 
                                             "[nm]", convert_in, iconvert_in)
        
        # Modules may additionally define a dummy "__SHARED__" layer for systems in which
        # multiple layers have identical lists of params and outputs 
        # (see report_shared_params() for a programmatic def and the PN_Junction
        # module for a physical example).
        # The notebook handles these in a special way and will look for this
        # entry in self.layers instead of each individual layer's entry.
        
        # Modules which do not share params and outputs should instead leave
        # __SHARED__ as None.
        
        # Models with only one layer should also leave __SHARED__ as None.
        
        self.shared_layer = None
        
        # Modules may also declare themselves as LGC-eligible, which means they
        # are compatible with the laser generation initial condition that sets
        # values for parameters named delta_N and delta_P.
        
        self.is_LGC_eligible = False
        

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
                    logger.info("Warning: Step size too large to resolve initial condition accurately")

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
    
    def report_shared_params(self):
        """
        A shared parameter is one that is tracked in every layer defined by a module.
        An example would be mu_N in the PN-junction, in which the n-type, buffer, and p-type
        layers all have this parameter defined.
        
        This function returns a list of all such shared parameters.
        
        For consistency, this and related functions should be the only methods
        used to check whether shared values exist.

        Returns
        -------
        set
            List of shared parameters (see above definition).

        """
        if len(self.layers) == 1:
            return set()
        
        return set.intersection(*[set(self.layers[layer].params.keys())
                                          for layer in self.layers])
    
    def report_shared_s_outputs(self):
        """
        Same as report_shared_params, but for s_outputs.

        """
        if len(self.layers) == 1:
            return set()
        
        return set.intersection(*[set(self.layers[layer].s_outputs.keys())
                                          for layer in self.layers])
    
    def report_shared_c_outputs(self):
        """
        Same as report_shared_params, but for c_outputs.

        """
        if len(self.layers) == 1:
            return set()
        
        return set.intersection(*[set(self.layers[layer].c_outputs.keys())
                                          for layer in self.layers])
    
    def report_shared_outputs(self):
        return set.union(*[self.report_shared_s_outputs(), self.report_shared_c_outputs()])
        
    def calc_inits(self):
        """
        Uses the self.param_dict to calculate initial conditions for ODEINT.
        
        In many cases this is just returning the appropriate parameter from layer's param dict.

        Returns
        -------
        dict {"param name":1D numpy array}
            Collection of initial condition arrays.

        """
        
        return {}
    
    def simulate(self, data_path, m, n, dt, flags, hmax_, init_conditions):
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
        flags : dict {"Flag internal name": Flag()}
            Collection of flag instances.
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
        data_dict : dict {"layer_name" :{"output name": 1D or 2D numpy array}}
            Collection of output data. 1D array if calculated once (e.g. an integral over space) and 2D if calculated at multiple times
            First dimension is time, second dimension is space
            Must return one dict per layer, each dict with one entry per 
            simulated or calculated output defined in the layer.
            
            Shared outputs (see report_shared_params()) should go into an additional
            dict with "layer_name" = "__SHARED__".
            See the PN-Junction module for an example of this.
        """
        data_dict = {}

        return data_dict
    
    def prep_dataset(self, datatype, target_layer, sim_data, params, flags, for_integrate=False, i=0, j=0, nen=False, extra_data = None):
        """
        Use the raw data in sim_data to calculate quantities in self.outputs_dict.
        If datatype is in self.simulated_outputs dict this just needs to return the correct item from sim_data

        Parameters
        ----------
        datatype : str
            The item we need to calculate.
        target_layer : str
            The name of the layer "datatype" is found in.
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
    
    def get_timeseries(self, pathname, datatype, parent_data, total_time, dt):
        """
        Calculate all time series associated with datatype after integrating.

        Parameters
        ----------
        pathname : str
            Path to simulated data .h5 files, excluding the datatype identifier.
            Use a string such as pathname + "-N.h5" to access the simulated data.
        datatype : str
            Data type that was just integrated. Use this value to associate different
            time series with related data types.
        parent_data : np.ndarray
            Values that were just integrated. Often the time series can be calculated
            directly from this if a good choice of datatype-time series
            association is made.
        total_time : float
            Final time value the simulated data go up to.
        dt : float
            Time step size.

        Returns
        -------
        list
            DESCRIPTION.

        """
        
        if datatype == "PL":
            return [("tau_diff", None)]
        
        elif datatype == "Another data type":
            return [("a different time series", None)]
        
        else:
            return
    
    def get_IC_regen(self, sim_data, param_dict, include_flags, grid_x):
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
        logger.info("Verifying selected module...")
        
        errors = ["{} verification error:".format(self.system_ID)]
        if not "symmetric_system" in self.flags_dict:
            logger.info("Warning: no symmetric_system flag defined."
                  "Automatically setting symmetric_system to FALSE")
            self.flags_dict["symmetric_system"] = ("Symmetric System", 0, 0)
        
        ## TODO: Add more as needed
        for layer in self.layers:
            for param in self.layers[layer].params:
                if '-' in param:
                    errors.append("'-' not allowed in param names")
                    break
                if ':' in param:
                    errors.append("':' not allowed in param names")
                    break
                
            for param in self.layers[layer].outputs:
                if ':' in param:
                    errors.append("':' not allowed in output names")
                    break
                
        if len(errors) > 1:
            logger.info("\n".join(errors))
            raise NotImplementedError
        return

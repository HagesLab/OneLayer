# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:49:49 2021

@author: cfai2
"""
from numpy import inf

class Layer:
    """ A space-discretized representation of a unique physical component.
    
    A model system is comprised of one or several Layers.
    Each Layer maintains its own space grid,
    tracks an internal set of state variables over its grid,
    may be characterized by different processes / equations / rules,
    and may interact with other layers.
    """
    def __init__(self, name, params, s_outputs, c_outputs, length_unit, convert_in):
        """
        Parameters
        ----------
        name: str
            Display name for this Layer.
        params: dict {"parameter name":Parameter}
            Permament state values associated with the Layer.
            
        s_outputs: dict {"output name":Output}
            Simulated outputs (state variables) associated with the Layer.
            Values which are directly timestepped forward by the simulation.
            
        c_outputs: dict {"output name":Output}
            Calculated outputs associated with the Layer.
            Secondary values which are calculated using simulated outputs.
            
        length_unit: str
            The length unit the space grid is assumed to be in.
            convert_in and convert_out should be used to ensure
            that units of params and outputs agree with this.
            
        convert_in: dict {"variable name":float}
            Table of values to unit-convert from GUI inputs to internal values 
            for solver.
        """
        self.name = name
        self.params = params
        self.param_count = len(params)
        self.s_outputs = s_outputs
        self.c_outputs = c_outputs
        self.outputs = {**self.s_outputs, **self.c_outputs}
        
        self.s_outputs_count = len(self.s_outputs)
        self.c_outputs_count = len(self.c_outputs)
        self.outputs_count = self.s_outputs_count + self.c_outputs_count
        
        self.total_length = -1
        self.dx = -1
        self.length_unit = length_unit
        self.grid_x_nodes = -1
        self.grid_x_edges = -1
        self.spacegrid_is_set = False
        
        self.convert_in = convert_in
        
        self.convert_out = {}
        for param in self.convert_in:
            self.convert_out[param] = self.convert_in[param] ** -1
            
        assert isinstance(params, dict), "Layer {} did not receive a dict of params".format(self.name)
        for param in self.params:
            assert isinstance(param, str), "Invalid param name {} in Layer {}".format(param, self.name)
            assert isinstance(self.params[param], Parameter), "Invalid param object for param {} in Layer {}".format(param, self.name)
        
        
        assert isinstance(self.outputs, dict), "Layer {} did not receive a dict of simulated outputs".format(self.name)
        for output in self.outputs:
            assert isinstance(output, str), "Invalid output name {} in Layer {}".format(param, self.name)
            assert isinstance(self.outputs[output], Output), "Invalid output object for output {} in Layer {}".format(param, self.name)
        
        return

class Characteristic:
    def __init__(self, units, is_edge):
        """ Base class for module variables.
        
        Parameters
        ----------
        units : str
            Units to be displayed on GUI. These can be anything as long as the correct conversions
            are in module's conversion dict, but labeling with common units are recommended for convenience.
        is_edge : bool
            Whether this quantity should be calculated at node edges or node centres.


        """
        self.units = units
        self.is_edge = is_edge
        return

class Parameter(Characteristic):
    def __init__(self, units, is_edge, valid_range=(-inf, inf), is_space_dependent=True):
        """ Helper class to store info about each of a Module's parameters
        and their initial distributions.
        
        Each Layer of a Module maintains a list of Parameters.
        Parameters do not change over time.

        Parameters
        ----------
        is_space_dependent : bool, optional
            Whether the GUI should allow assigning a space-dependent distribution
            (i.e. an array of node values) to this parameter.
            
            Usually this should be true, but some parameters involving boundary
            conditions, like surface recombination rates, should always
            be single-valued and thus have this as False.
            
            The default is True.

        Returns
        -------
        None.

        """
        super().__init__(units, is_edge)

        self.value = 0
        assert isinstance(valid_range, tuple), "A tuple was expected for Parameter valid range"
        # TODO: Implement value verification using this
        self.valid_range = valid_range # Min, max
        self.is_space_dependent = is_space_dependent
        self.param_rules = []
        return
    
class Output(Characteristic):

    def __init__(self, display_name, units, xlabel, xvar, is_edge, analysis_plotable=True, yscale='symlog', yfactors=(1,1)):
        """ Helper class for managing info about each of a Module's output values.
        
        Each Layer tracks a list of Outputs.
        Outputs have initial values and change over time.
        
        Parameters
        ----------
        display_name : strs
            What the GUI should list this item as. Can be different from internal keys used by module outputs_dict.
        xlabel : str
            Unit to be printed on horizontal plot axes.
        xvar : str, either "position" or "time"
            Used by GUI plot_overview_analysis to plot either with a time grid or a space grid.
        is_integrated : bool, optional
            Whether this value is calculated using an integration procedure.
            Defined for convenience in module's plot_overview_analysis.
            The default is False.
        yscale : str, optional, must be a valid matplotlib plot scale
            Which plot scale the sim and overview plotters should use e.g. linear or log. The default is 'symlog'.
            Detailed analysis plots have an experimental autoscaling function.
        yfactors : tuple(float, float), optional
            Plot axis range for sim and overview plotters. These plot from min_value * yfactors[0] to max_value * yfactors[1].
            The default is (1,1).

        Returns
        -------
        None.

        """
        super().__init__(units, is_edge)
        self.display_name = display_name
        self.xlabel = xlabel
        self.xvar = xvar
        self.analysis_plotable = analysis_plotable
        self.yscale = yscale
        self.yfactors = yfactors
        return    
    
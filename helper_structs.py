# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:49:49 2021

@author: cfai2
"""

class Characteristic:
    def __init__(self, units, is_edge):
        """
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
    # Helper class to store info about each of a Nanowire's parameters and initial distributions
    def __init__(self, units, is_edge):
        super().__init__(units, is_edge)

        self.value = 0
        
        # TODO: Implement value verification using this
        self.valid_range = (None, None) # Min, max
        
        self.param_rules = []
        return
    
class Output(Characteristic):
    
    def __init__(self, display_name, units, xlabel, xvar, is_edge, calc_func=None, is_integrated=False, analysis_plotable=True, yscale='symlog', yfactors=(1,1)):
        """
        Parameters
        ----------
        display_name : str
            What the GUI should list this item as. Can be different from internal keys used by module outputs_dict.
        xlabel : str
            Unit to be printed on horizontal plot axes.
        xvar : str, either "position" or "time"
            Used by GUI plot_overview_analysis to plot either with a time grid or a space grid.
        calc_func : function, optional
            finite.py function used to calculate this value, if applicable. 
            Defined for convenience in module's plot_overview_analysis.
            The default is None.
        is_integrated : bool, optional
            Whether this value is calculated using an integration procedure.
            Defined for convenience in module's plot_overview_analysis.
            The default is False.
        analysis_plotable : bool, optional
            Whether this output should be selectable from the detailed analysis tab. 
            This is useful for if there are supplementary calculations desired while calculating this output,
            but these should not be asked for by themselves. See nanowire tau_diff for an example.
            The default is True.
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
        self.is_integrated = is_integrated
        self.analysis_plotable = analysis_plotable # Whether this Output can be selected from analysis tab's plot feature
        self.yscale = yscale
        self.yfactors = yfactors
        self.calc_func = calc_func
        return    
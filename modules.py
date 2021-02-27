# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:47:55 2021

@author: cfai2
"""
import numpy as np
import finite
from helper_structs import Parameter, Output
from utils import u_read, to_index, to_array, to_pos
import tables

def mod_list():
    return {"Nanowire":Nanowire, "Neumann Bound Heatplate":HeatPlate}

class OneD_Model:
    
    def __init__(self):
        self.system_ID = "INSERT MODEL NAME HERE"
        self.total_length = -1
        self.dx = -1
        self.length_unit = "INSERT LENGTH UNIT HERE"
        self.grid_x_nodes = -1
        self.grid_x_edges = -1
        self.spacegrid_is_set = False
        
        self.param_dict = {}
        self.param_count = len(self.param_dict)
        
        # This tells TEDs what flags there will be and their 'layman's name' shown on the interface as,
        # but we actually don't need to store the flag values here
        # The reason why is that info is already stored in whether a flag appears visually selected on the interface
        self.flags_dict = {"Flag1":"Flag1's name"}
        
        # List of all variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        self.simulation_outputs_dict = {"Output1":Output("y", units="[hamburgers/football field]", xlabel="a.u.", xvar="position", is_edge=False, is_calculated=False,is_integrated=False, yscale='log', yfactors=(1e-4,1e1))}
        
        # List of all variables calculated from those in simulation_outputs_dict
        self.calculated_outputs_dict = {"deltaN":Output("delta_N", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False, is_calculated=True, calc_func=finite.delta_n, is_integrated=False)}
        
        self.outputs_dict = {**self.simulation_outputs_dict, **self.calculated_outputs_dict}
        
        self.simulation_outputs_count = len(self.simulation_outputs_dict)
        self.calculated_outputs_count = len(self.calculated_outputs_dict)
        self.total_outputs_count = self.simulation_outputs_count + self.calculated_outputs_count
        ## Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        self.convert_in_dict = {"Output1": ((1e-2) / (1e2)),               # [hamburgers/football field] to [centihamburgers/yard]
                                "deltaN": ((1e-7) ** 3)                    # [cm^-3] to [nm^-3]
                                }
        
        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
        self.convert_out_dict = {}
        for param in self.convert_in_dict:
            self.convert_out_dict[param] = self.convert_in_dict[param] ** -1

        return
    
    
    def add_param_rule(self, param_name, new_rule):
        self.param_dict[param_name].param_rules.append(new_rule)
        self.update_param_toarray(param_name)
        return

    def swap_param_rules(self, param_name, i):
        self.param_dict[param_name].param_rules[i], self.param_dict[param_name].param_rules[i-1] = self.param_dict[param_name].param_rules[i-1], self.param_dict[param_name].param_rules[i]
        self.update_param_toarray(param_name)
        return

    def remove_param_rule(self, param_name, i):
        self.param_dict[param_name].param_rules.pop(i)
        self.update_param_toarray(param_name)
        return
    
    def removeall_param_rules(self, param_name):
        self.param_dict[param_name].param_rules = []
        self.param_dict[param_name].value = 0
        return
    
    def update_param_toarray(self, param_name):
        # Recalculate a Parameter from its Param_Rules
        # This should be done every time the Param_Rules are changed
        param = self.param_dict[param_name]

        if param.is_edge:
            new_param_value = np.zeros(self.grid_x_edges.__len__())
        else:
            new_param_value = np.zeros(self.grid_x_nodes.__len__())

        for condition in param.param_rules:
            i = to_index(condition.l_bound, self.dx, self.total_length, param.is_edge)
            
            # If the left bound coordinate exceeds the width of the node toIndex() (which always rounds down) 
            # assigned, it should actually be mapped to the next node
            if (condition.l_bound - to_pos(i, self.dx, param.is_edge) > self.dx / 2): i += 1

            if (condition.type == "POINT"):
                new_param_value[i] = condition.l_boundval

            elif (condition.type == "FILL"):
                j = to_index(condition.r_bound, self.dx, self.total_length, param.is_edge)
                new_param_value[i:j+1] = condition.l_boundval

            elif (condition.type == "LINE"):
                slope = (condition.r_boundval - condition.l_boundval) / (condition.r_bound - condition.l_bound)
                j = to_index(condition.r_bound, self.dx, self.total_length, param.is_edge)

                ndx = np.linspace(0, self.dx * (j - i), j - i + 1)
                new_param_value[i:j+1] = condition.l_boundval + ndx * slope

            elif (condition.type == "EXP"):
                j = to_index(condition.r_bound, self.dx, self.total_length, param.is_edge)

                ndx = np.linspace(0, j - i, j - i + 1)
                try:
                    new_param_value[i:j+1] = condition.l_boundval * np.power(condition.r_boundval / condition.l_boundval, ndx / (j - i))
                except FloatingPointError:
                    print("Warning: Step size too large to resolve initial condition accurately")

        param.value = new_param_value
        return
    
    def DEBUG_print(self):
        #print("Behold the One Nanowire in its infinite glory:")
        mssg = ""
        if self.spacegrid_is_set:
            mssg += ("Space Grid is set\n")
            mssg += ("Nodes: {}\n".format(self.grid_x_nodes))
            mssg += ("Edges: {}\n".format(self.grid_x_edges))
        else:
            mssg += ("Grid is not set\n")

        for param in self.param_dict:
            mssg += ("{} {}: {}\n".format(param, self.param_dict[param].units, self.param_dict[param].value))

        return mssg
    
    def calc_inits(self):
        # Do stuff here
        return {}
    
    def simulate(self, data_path, m, n, dt, params, flags, do_ss, hmax_, init_conditions, write_output):
        # No strict rules on how simulate() needs to look - as long as it calls the appropriate ode() from finite.py with the correct args
        return None
    
    def get_overview_analysis(self, params, tsteps, data_dirname, file_name_base):
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        data_dict = {}
        
        # Insert calculations and stuff here
        
        
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, do_plot_override=True, i=0, j=0, nen=False, extra_data = None):
        data = None
        
        # data = (your calcs here)
                
        return data
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        param_dict["Output1"] = None
        return
    
    def verify(self):
        print("Verifying selected module...")
        params = set(self.param_dict.keys()).union(set(self.outputs_dict.keys()))
        params_in_cdict = set(self.convert_in_dict.keys())
        assert (params.issubset(params_in_cdict)), "Error: conversion_dict is missing entries {}".format(params.difference(params_in_cdict))
        return

class Nanowire(OneD_Model):
    # A Nanowire object stores all information regarding the initial state being edited in the IC tab
    # And functions for managing other previously simulated nanowire data as they are loaded in
    def __init__(self):
        super().__init__()
        self.system_ID = "Nanowire"
        self.length_unit = "[nm]"
        self.param_dict = {"Mu_N":Parameter(units="[cm^2 / V s]", is_edge=True), "Mu_P":Parameter(units="[cm^2 / V s]", is_edge=True), 
                            "N0":Parameter(units="[cm^-3]", is_edge=False), "P0":Parameter(units="[cm^-3]", is_edge=False), 
                            "B":Parameter(units="[cm^3 / s]", is_edge=False), "Tau_N":Parameter(units="[ns]", is_edge=False), 
                            "Tau_P":Parameter(units="[ns]", is_edge=False), "Sf":Parameter(units="[cm / s]", is_edge=False), 
                            "Sb":Parameter(units="[cm / s]", is_edge=False), "Temperature":Parameter(units="[K]", is_edge=True), 
                            "Rel-Permitivity":Parameter(units="", is_edge=True), "Ext_E-Field":Parameter(units="[V/um]", is_edge=True),
                            "Theta":Parameter(units="[cm^-1]", is_edge=False), "Alpha":Parameter(units="[cm^-1]", is_edge=False), 
                            "Delta":Parameter(units="", is_edge=False), "Frac-Emitted":Parameter(units="", is_edge=False),
                            "deltaN":Parameter(units="[cm^-3]", is_edge=False), "deltaP":Parameter(units="[cm^-3]", is_edge=False), 
                            "E_field":Parameter(units="[WIP]", is_edge=True), "Ec":Parameter(units="[WIP]", is_edge=True),
                            "electron_affinity":Parameter(units="[WIP]", is_edge=True)}
        

        self.param_count = len(self.param_dict)
        
        self.flags_dict = {"ignore_alpha":"Ignore Photon Recycle",
                           "symmetric_system":"Symmetric System"}

        # List of all variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        self.simulation_outputs_dict = {"N":Output("N", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False, is_calculated=False,is_integrated=False, yscale='symlog', yfactors=(1e-4,1e1)), 
                                        "P":Output("P", units="[cm^-3]", xlabel="nm", xvar="position",is_edge=False, is_calculated=False,is_integrated=False, yscale='symlog', yfactors=(1e-4,1e1)), 
                                        "E_field":Output("Electric Field", units="[WIP]", xlabel="nm", xvar="position",is_edge=True, is_calculated=False,is_integrated=False, yscale='linear')}
        
        # List of all variables calculated from those in simulation_outputs_dict
        self.calculated_outputs_dict = {"deltaN":Output("delta_N", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False, is_calculated=True, calc_func=finite.delta_n, is_integrated=False),
                                         "deltaP":Output("delta_P", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False, is_calculated=True, calc_func=finite.delta_p, is_integrated=False),
                                         "RR":Output("Radiative Recombination", units="[cm^-3 s^-1]", xlabel="nm", xvar="position",is_edge=False, is_calculated=True, calc_func=finite.radiative_recombination, is_integrated=False),
                                         "NRR":Output("Non-radiative Recombination", units="[cm^-3 s^-1]", xlabel="nm", xvar="position", is_edge=False, is_calculated=True, calc_func=finite.nonradiative_recombination, is_integrated=False),
                                         "PL":Output("TRPL", units="[WIP]", xlabel="ns", xvar="time", is_edge=False, is_calculated=True, calc_func=finite.new_integrate, is_integrated=True),
                                         "tau_diff":Output("-(dln(TRPL)/dt)^-1", units="[WIP]", xlabel="ns", xvar="time", is_edge=False, is_calculated=True, calc_func=finite.tau_diff, is_integrated=True, analysis_plotable=False)}
        
        self.outputs_dict = {**self.simulation_outputs_dict, **self.calculated_outputs_dict}
        
        self.simulation_outputs_count = len(self.simulation_outputs_dict)
        self.calculated_outputs_count = len(self.calculated_outputs_dict)
        self.total_outputs_count = self.simulation_outputs_count + self.calculated_outputs_count
        ## Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        self.convert_in_dict = {"Mu_N": ((1e7) ** 2) / (1e9), "Mu_P": ((1e7) ** 2) / (1e9), # [cm^2 / V s] to [nm^2 / V ns]
                                "N0": ((1e-7) ** 3), "P0": ((1e-7) ** 3),                   # [cm^-3] to [nm^-3]
                                "Thickness": 1, "dx": 1,
                                "B": ((1e7) ** 3) / (1e9),                                  # [cm^3 / s] to [nm^3 / ns]
                                "Tau_N": 1, "Tau_P": 1,                                     # [ns]
                                "Sf": (1e7) / (1e9), "Sb": (1e7) / (1e9),                   # [cm / s] to [nm / ns]
                                "Temperature": 1, "Rel-Permitivity": 1, 
                                "Ext_E-Field": 1e-3,                                        # [V/um] to [V/nm]
                                "Theta": 1e-7, "Alpha": 1e-7,                               # [cm^-1] to [nm^-1]
                                "Delta": 1, "Frac-Emitted": 1,
                                "deltaN": ((1e-7) ** 3), "deltaP": ((1e-7) ** 3),
                                "Ec": 1, "electron_affinity": 1,
                                "N": ((1e-7) ** 3), "P": ((1e-7) ** 3),                     # [cm^-3] to [nm^-3]
                                "E_field": 1, 
                                "tau_diff": 1}
        
        # FIXME: Check these
        self.convert_in_dict["RR"] = self.convert_in_dict["B"] * self.convert_in_dict["N"] * self.convert_in_dict["P"]
        self.convert_in_dict["NRR"] = self.convert_in_dict["N"] / self.convert_in_dict["Tau_N"]
        self.convert_in_dict["PL"] = self.convert_in_dict["RR"] * self.convert_in_dict["Theta"]
        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
        self.convert_out_dict = {}
        for param in self.convert_in_dict:
            self.convert_out_dict[param] = self.convert_in_dict[param] ** -1

        return
    
    def calc_inits(self):
        # The three true initial conditions for our three coupled ODEs
        # N = N0 + delta_N
        init_N = (self.param_dict["N0"].value + self.param_dict["deltaN"].value) * self.convert_in_dict["N"]
        init_P = (self.param_dict["P0"].value + self.param_dict["deltaP"].value) * self.convert_in_dict["P"]
        init_E_field = self.param_dict["E_field"].value * self.convert_in_dict["E_field"]
        # "Typecast" single values to uniform arrays
        if not isinstance(init_N, np.ndarray):
            init_N = np.ones(self.grid_x_nodes.__len__()) * init_N
            
        if not isinstance(init_P, np.ndarray):
            init_P = np.ones(self.grid_x_nodes.__len__()) * init_P
            
        if not isinstance(init_E_field, np.ndarray):
            init_E_field = np.ones(self.grid_x_edges.__len__()) * init_E_field
        
        return {"N":init_N, "P":init_P, "E_field":init_E_field}
    
    def simulate(self, data_path, m, n, dt, params, flags, do_ss, hmax_, init_conditions, write_output):
        # No strict rules on how simulate() needs to look - as long as it calls the appropriate ode() from finite.py with the correct args
        return finite.ode_nanowire(data_path, m, n, self.dx, dt, params,
                                    not flags['ignore_alpha'].value(), flags['symmetric_system'].value(), do_ss, hmax_, write_output,
                                    init_conditions["N"], init_conditions["P"], init_conditions["E_field"])
    
    def get_overview_analysis(self, params, tsteps, data_dirname, file_name_base):
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        data_dict = {}
        
        for raw_output_name in self.simulation_outputs_dict:
            data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, raw_output_name)
            data = []
            for tstep in tsteps:
                data.append(u_read(data_filename, t0=tstep, single_tstep=True))
            
            data_dict[raw_output_name] = np.array(data)
            
        for calculated_output_name, output_obj in self.calculated_outputs_dict.items():
            if not output_obj.is_integrated:
                data_dict[calculated_output_name] = output_obj.calc_func(data_dict, params)
                
                
        with tables.open_file(data_dirname + "\\" + file_name_base + "-n.h5", mode='r') as ifstream_N, \
            tables.open_file(data_dirname + "\\" + file_name_base + "-p.h5", mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data)
            temp_P = np.array(ifstream_P.root.data)
        temp_RR = finite.radiative_recombination({"N":temp_N, "P":temp_P}, params)
        PL_base = finite.prep_PL(temp_RR, 0, to_index(params["Total_length"], params["Node_width"], params["Total_length"]), False, params)
        data_dict["PL"] = self.calculated_outputs_dict["PL"].calc_func(PL_base, 0, params["Total_length"], params["Node_width"], params["Total_length"], 
                                                                       False)
        
        data_dict["tau_diff"] = self.calculated_outputs_dict["tau_diff"].calc_func(data_dict["PL"], params["dt"])
        
        for data in data_dict:
            data_dict[data] *= self.convert_out_dict[data]
            
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, do_plot_override=True, i=0, j=0, nen=False, extra_data = None):
        # For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        if (datatype in self.simulation_outputs_dict):
            data = sim_data[datatype]
        
        else:
            if (datatype == "deltaN"):
                data = finite.delta_n(sim_data, params)
                
            elif (datatype == "deltaP"):
                data = finite.delta_p(sim_data, params)
                
            elif (datatype == "RR"):
                data = finite.radiative_recombination(sim_data, params)

            elif (datatype == "NRR"):
                data = finite.nonradiative_recombination(sim_data, params)

            elif (datatype == "PL"):
    
                if do_plot_override:
                    rad_rec = finite.radiative_recombination(sim_data, params)
                    data = finite.prep_PL(rad_rec, 0, len(rad_rec), need_extra_node=False, params=params).flatten()
                else:
                    rad_rec = finite.radiative_recombination(extra_data, params)
                    data = finite.prep_PL(rad_rec, i, j, nen, params)
            else:
                raise ValueError
                
        return data
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        param_dict["deltaN"] = (sim_data["N"] - param_dict["N0"]) if include_flags['N'] else np.zeros(grid_x.__len__())
                    
        param_dict["deltaP"] = (sim_data["P"] - param_dict["P0"]) if include_flags['P'] else np.zeros(grid_x.__len__())
        
        param_dict["E_field"] = sim_data["E_field"] if include_flags['E_field'] else np.zeros(grid_x.__len__() + 1)

        return
    
class HeatPlate(OneD_Model):
    
    def __init__(self):
        super().__init__()
        self.system_ID = "HeatPlate (Const Bound Flux)"
        self.length_unit = "[m]"
        
        self.param_dict = {"k":Parameter(units="[W / m k]", is_edge=False), "Cp":Parameter(units="[J / kg K]", is_edge=False), 
                            "density":Parameter(units="[kg m^-3]", is_edge=False), "init_T":Parameter(units="[K]", is_edge=False),
                            "Left_flux":Parameter(units="[W m^-2]", is_edge=False), "Right_flux":Parameter(units="[W m^-2]", is_edge=False)}
        

        self.param_count = len(self.param_dict)
        
        self.flags_dict = {"symmetric_system":"Symmetric System"}

        # List of all variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        self.simulation_outputs_dict = {"T":Output("Temperature", units="[K]", xlabel="m", xvar="position", is_edge=False, is_calculated=False,is_integrated=False, yscale='linear')}
        
        # List of all variables calculated from those in simulation_outputs_dict
        self.calculated_outputs_dict = {"q":Output("Heat Flux", units="[W/m^2]", xlabel="m", xvar="position", is_edge=True, is_calculated=True, calc_func=finite.heatflux, is_integrated=False)}
        
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
        init_T = self.param_dict['init_T'].value * self.convert_in_dict["T"]
        init_T = to_array(init_T, len(self.grid_x_nodes), False)
        return {"T":init_T}
    
    def simulate(self, data_path, m, n, dt, params, flags, do_ss, hmax_, init_conditions, write_output):
        # No strict rules on how simulate() needs to look - as long as it calls the appropriate ode() from finite.py with the correct args
        return finite.ode_heatplate(data_path, m, n, self.dx, dt, params,
                                    write_output)
    
    def get_overview_analysis(self, params, tsteps, data_dirname, file_name_base):
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        data_dict = {}
        
        for raw_output_name in self.simulation_outputs_dict:
            data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, raw_output_name)
            data = []
            for tstep in tsteps:
                data.append(u_read(data_filename, t0=tstep, single_tstep=True))
            
            data_dict[raw_output_name] = np.array(data)
            
        for calculated_output_name, output_obj in self.calculated_outputs_dict.items():
            if not output_obj.is_integrated:
                data_dict[calculated_output_name] = output_obj.calc_func(data_dict, params)
                
                
        for data in data_dict:
            data_dict[data] *= self.convert_out_dict[data]
            
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, do_plot_override=True, i=0, j=0, nen=False, extra_data = None):
        # For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        if (datatype in self.simulation_outputs_dict):
            data = sim_data[datatype]
        
        else:
            if (datatype == "q"):
                data = finite.heatflux(sim_data, params)

            else:
                raise ValueError
                
        return data
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        param_dict["T"] = sim_data["T"] if include_flags['T'] else np.zeros(grid_x.__len__())

        return
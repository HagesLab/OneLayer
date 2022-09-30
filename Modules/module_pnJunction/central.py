# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
from _OneD_Model import OneD_Model
from Modules.module_pnJunction.definitions import define_layers
from Modules.module_pnJunction.definitions import define_flags
from Modules.module_pnJunction.initializations import PN_Junction_Initial_Conditions
from Modules.module_pnJunction.analysis import submodule_get_overview_analysis
from Modules.module_pnJunction.analysis import submodule_prep_dataset
from Modules.module_pnJunction.analysis import submodule_get_timeseries
from Modules.module_pnJunction.analysis import submodule_get_IC_carry
from Modules.module_pnJunction.simulations import OdeTwoLayerSimulation



q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]


class PN_Junction(OneD_Model):
    # A Nanowire object stores all information regarding the initial state being edited in the IC tab
    # And functions for managing other previously simulated nanowire data as they are loaded in
    def __init__(self):
        super().__init__()
        self.system_ID = "PN_Junction"
        self.time_unit = "[ns]"
        self.flags_dict = define_flags()
        self.layers = define_layers()

        return


    def calc_inits(self):
        """Calculate initial electron and hole density distribution"""
        
        ntype = self.layers["N-type"]
        buffer = self.layers["buffer"]
        ptype = self.layers["P-type"]

        pnjunction_inits = PN_Junction_Initial_Conditions(ntype, buffer, ptype)

        return pnjunction_inits.format_inits_to_dict()


    def simulate(self, data_path, m, n, dt, flags, hmax_, init_conditions):
        """Calls ODEINT solver."""
        mapi = self.layers["MAPI"]
        rubrene = self.layers["Rubrene"]
        for param_name, param in mapi.params.items():
            param.value *= mapi.convert_in[param_name]
            
        for param_name, param in rubrene.params.items():
            param.value *= rubrene.convert_in[param_name]

        ode_two_layer = OdeTwoLayerSimulation(mapi, rubrene, m, flags, init_conditions)
        ode_two_layer.simulate(data_path, n, dt, hmax_)


    def get_overview_analysis(self, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        data_dict = submodule_get_overview_analysis(self.layers, params, flags, total_time, dt, tsteps, data_dirname, file_name_base)
        return data_dict


    def prep_dataset(self, datatype, sim_data, params, flags, for_integrate=False,
                     i=0, j=0, nen=False, extra_data=None):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        where_layer = self.find_layer(datatype)
        layer = self.layers[where_layer]
        data = submodule_prep_dataset(where_layer, layer, datatype, sim_data, params,
                    for_integrate, i, j, nen, extra_data)
        return data


    def get_timeseries(self, pathname, datatype, parent_data, total_time, dt, params, flags):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        timeseries = submodule_get_timeseries(pathname, datatype, parent_data, total_time, dt, params, flags)
        return timeseries

    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        carry = submodule_get_IC_carry(sim_data, param_dict, include_flags, grid_x)
        return carry

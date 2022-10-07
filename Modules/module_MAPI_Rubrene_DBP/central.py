# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
from _OneD_Model import OneD_Model
from Modules.module_MAPI_Rubrene_DBP.definitions import define_layers
from Modules.module_MAPI_Rubrene_DBP.definitions import define_flags
from Modules.module_MAPI_Rubrene_DBP.initializations import MAPI_Rubrene_Initial_Conditions
from Modules.module_MAPI_Rubrene_DBP.analysis import submodule_get_overview_analysis
from Modules.module_MAPI_Rubrene_DBP.analysis import submodule_prep_dataset
from Modules.module_MAPI_Rubrene_DBP.analysis import submodule_get_timeseries
from Modules.module_MAPI_Rubrene_DBP.analysis import submodule_get_IC_regen
from Modules.module_MAPI_Rubrene_DBP.simulations import OdeTwoLayerSimulation



q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]


class MAPI_Rubrene(OneD_Model):
    # A Nanowire object stores all information regarding the initial state being edited in the IC tab
    # And functions for managing other previously simulated nanowire data as they are loaded in
    def __init__(self):
        super().__init__()
        self.system_ID = "MAPI_Rubrene"
        self.time_unit = "[ns]"
        self.flags_dict = define_flags()
        self.layers = define_layers()

        return


    def calc_inits(self):
        """Calculate initial electron and hole density distribution"""
        
        mapi = self.layers["MAPI"]
        rubrene = self.layers["Rubrene"]

        mapi_rubrene_inits = MAPI_Rubrene_Initial_Conditions(mapi, rubrene)

        return mapi_rubrene_inits.format_inits_to_dict()


    def simulate(self, data_path, m, n, dt, flags, hmax_, rtol_, atol_, init_conditions):
        """Calls ODEINT solver."""
        mapi = self.layers["MAPI"]
        rubrene = self.layers["Rubrene"]
        for param_name, param in mapi.params.items():
            param.value *= mapi.convert_in[param_name]
            
        for param_name, param in rubrene.params.items():
            param.value *= rubrene.convert_in[param_name]

        ode_two_layer = OdeTwoLayerSimulation(mapi, rubrene, m, flags, init_conditions)
        ode_two_layer.simulate(data_path, n, dt, hmax_, rtol_, atol_)


    def get_overview_analysis(self, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        data_dict = submodule_get_overview_analysis(self.layers, params, flags, total_time, dt, tsteps, data_dirname, file_name_base)
        return data_dict


    def prep_dataset(self, datatype, target_layer, sim_data, params, flags, for_integrate=False,
                     i=0, j=0, nen=False, extra_data=None):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        layer = self.layers[target_layer]
        data = submodule_prep_dataset(target_layer, layer, datatype, sim_data, params,
                    for_integrate, i, j, nen, extra_data)
        return data


    def get_timeseries(self, pathname, datatype, parent_data, total_time, dt, params, flags):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        timeseries = submodule_get_timeseries(pathname, datatype, parent_data, total_time, dt, params, flags)
        return timeseries

    
    def get_IC_regen(self, sim_data, param_dict, include_flags, grid_x):
        """Dispatched all logic to a submodule while keeping contract
        (name and arguments of method) with rest of the system for stability"""
        regen = submodule_get_IC_regen(sim_data, param_dict, include_flags, grid_x)
        return regen

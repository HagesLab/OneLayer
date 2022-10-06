import numpy as np
import os
import tables
from scipy import integrate as intg
from io_utils import u_read
from utils import to_index, new_integrate
from Modules.module_pnJunction.calculations import radiative_recombination
from Modules.module_pnJunction.calculations import prep_PL
# from Modules.module_pnJunction.calculations import tau_diff
# from Modules.module_pnJunction.calculations import E_field_r
# from Modules.module_pnJunction.calculations import TTA
from Modules.module_pnJunction.calculations import CalculatedOutputs, tau_diff


def submodule_get_overview_analysis(layers, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
    """Calculates at a selection of sample times: N, P, (total carrier densities)
    delta_N, delta_P, (above-equilibrium carrier densities)
    internal electric field due to differences in N, P,
    radiative recombination,
    non-radiative (SRH model) recombination,
    
    Integrates over nanowire length: PL due to
    radiative recombination, waveguiding, and carrier regeneration.
    
    Since all outputs are shared in this module, we will plot all layers on single plots
    and integrals will be over all three layers.
    """


    data_dict = {"__SHARED__":{}}
    
    any_layer = next(iter(layers))
    any_layer = layers[any_layer]
    
    convert_out = any_layer.convert_out
    for raw_output_name in any_layer.s_outputs:
        
        data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, 
                                                raw_output_name)
        data = []
        for tstep in tsteps:
            data.append(u_read(data_filename, t0=tstep, single_tstep=True))
        
        data_dict["__SHARED__"][raw_output_name] = np.array(data)
                
    
    calculated_outputs = CalculatedOutputs(data_dict["__SHARED__"], params, any_layer)
    
    #data_dict["MAPI"]["E_field"] = calculated_outputs.E_field()
    data_dict["__SHARED__"]["delta_N"] = calculated_outputs.delta_n()
    data_dict["__SHARED__"]["delta_P"] = calculated_outputs.delta_p()
    
    data_dict["__SHARED__"]["RR"] = calculated_outputs.radiative_recombination()
    data_dict["__SHARED__"]["NRR"] = calculated_outputs.nonradiative_recombination()
    
    data_dict["__SHARED__"]["voltage"] = calculated_outputs.voltage()
    data_dict["__SHARED__"]["E_field"] = calculated_outputs.E_field()
          
    #### PL ####
    with tables.open_file(os.path.join(data_dirname, file_name_base + "-N.h5"), mode='r') as ifstream_N, \
        tables.open_file(os.path.join(data_dirname, file_name_base + "-P.h5"), mode='r') as ifstream_P:
        temp_N = np.array(ifstream_N.root.data)
        temp_P = np.array(ifstream_P.root.data)

    data_dict["__SHARED__"]["PL"] = calculated_outputs.PL(temp_N, temp_P, do_integrate=True)
    
    data_dict["__SHARED__"]["avg_delta_N"] = calculated_outputs.average_delta_n(temp_N)
    
    #try:
    data_dict["__SHARED__"]["tau_diff"] = tau_diff(data_dict["__SHARED__"]["PL"], dt)
    # except Exception:
    #     print("Error: failed to calculate tau_diff")
    #     data_dict["__SHARED__"]["tau_diff"] = 0
    #################

    for data in data_dict["__SHARED__"]:
        data_dict["__SHARED__"][data] *= convert_out[data]
        
    return data_dict


def submodule_prep_dataset(where_layer, layer, datatype, sim_data, params, for_integrate, 
                     i, j, nen, extra_data):
    """ Provides delta_N, delta_P, electric field,
    recombination, and spatial PL values on demand."""
    # For N, P this is just reading the data but for others we'll calculate it in situ
    layer_params = params[where_layer]
    layer_sim_data = sim_data[where_layer]
    data = None
    if (datatype in layer.s_outputs):
        data = layer_sim_data[datatype]
    
    else:
        calculated_outputs = CalculatedOutputs(sim_data[where_layer], params, layer)
        if (datatype == "delta_N"):
            data = calculated_outputs.delta_n()
            
        elif (datatype == "delta_P"):
            data = calculated_outputs.delta_p()
            
        elif (datatype == "delta_T"):
            data = calculated_outputs.delta_T()
            
        elif (datatype == "RR"):
            data = calculated_outputs.radiative_recombination()

        elif (datatype == "NRR"):
            data = calculated_outputs.nonradiative_recombination()
            
        elif (datatype == "voltage"):
            data = calculated_outputs.voltage()
            
        elif (datatype == "E_field"):
            data = calculated_outputs.E_field()

        elif (datatype == "PL"):

            if for_integrate:
                rad_rec = radiative_recombination(extra_data[where_layer], layer_params)
                data = prep_PL(rad_rec, i, j, nen, layer_params, where_layer)
            else:
                data = calculated_outputs.PL(sim_data[where_layer]['N'], sim_data[where_layer]['P'], do_integrate=False).flatten()
                        
        else:
            raise ValueError
            
    return data


def submodule_get_timeseries(pathname, datatype, parent_data, total_time, dt, params, flags):
    """Depending on requested datatype, returns the equivalent timeseries."""

    if datatype == "delta_N":
        temp_dN = parent_data / params["MAPI"]["Total_length"]
        return [("avg_delta_N", temp_dN)]
    
    if datatype == "mapi_PL":
        # photons emitted per photon absorbed
        # generally this is meaningful in ss mode - thus units are [phot/nm^2 ns] per [carr/nm^2 ns]
        with tables.open_file(pathname + "-N.h5", mode='r') as ifstream_N:
            temp_init_N = np.array(ifstream_N.root.data[0,:])
            
        temp_init_N = intg.trapz(temp_init_N, dx=params["MAPI"]["Node_width"])
        try:
            tdiff = tau_diff(parent_data, dt)
        except FloatingPointError:
            print("Error: failed to calculate tau_diff - effective lifetime is near infinite")
            tdiff = np.zeros_like(np.linspace(0, total_time, int(total_time/dt) + 1))
            
        return [("tau_diff", tdiff),
                ("eta_MAPI", parent_data / temp_init_N)]
                
    
    elif datatype == "TTA":
        with tables.open_file(pathname + "-N.h5", mode='r') as ifstream_N, \
            tables.open_file(pathname + "-P.h5", mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data[:,-1])
            temp_init_N = np.array(ifstream_N.root.data[0,:])
            temp_P = np.array(ifstream_P.root.data[:,-1])

        if flags.get("do_sct"):
            with tables.open_file(os.path.join(pathname + "-P_up.h5"), mode='r') as ifstream_Q:
                temp_P_up = np.array(ifstream_Q.root.data[:, 0])

        temp_init_N = intg.trapz(temp_init_N, dx=params["MAPI"]["Node_width"])

        tail_n0 = params["MAPI"]["N0"]
        if isinstance(tail_n0, np.ndarray):
            tail_n0 = tail_n0[-1]
            
        tail_p0 = params["MAPI"]["P0"]
        if isinstance(tail_p0, np.ndarray):
            tail_p0 = tail_p0[-1]

        if flags.get("no_upconverter"):
            t_form = temp_N * 0
        else:
            if flags.get("do_sct"):
                # TODO: Verify this is correct for the seq charge transfer
                t_form = params["Rubrene"]["Ssct"] * (temp_N * temp_P_up)
            else:
                t_form = params["Rubrene"]["St"] * ((temp_N * temp_P - tail_n0 * tail_p0)
                                        / (temp_N + temp_P))
        
        with np.errstate(invalid='ignore', divide='ignore'):
            # In order:
            # Triplets formed per photon absorbed
            t_form_eff = np.where(temp_init_N==0, 0, t_form / temp_init_N)
            # Singlets formed per triplet formed
            t_anni_eff = np.where(t_form==0, 0, parent_data / t_form)
            # Singlets formed per photon absorbed
            s_up_eff = t_form_eff * t_anni_eff
        return [("T_form_eff", t_form_eff),
                ("T_anni_eff", t_anni_eff),
                ("S_up_eff", s_up_eff)]
    
    elif datatype == "dbp_PL":
        with tables.open_file(pathname + "-N.h5", mode='r') as ifstream_N:
            temp_init_N = np.array(ifstream_N.root.data[0,:])
            
        temp_init_N = intg.trapz(temp_init_N, dx=params["MAPI"]["Node_width"])
        
        # DBP photons emitted per photon absorbed
        return [("eta_UC", parent_data / temp_init_N)]
    else:
        return


def submodule_get_IC_carry(sim_data, param_dict, include_flags, grid_x):
    """ Set delta_N and delta_P of outgoing regenerated IC file."""
    param_dict["MAPI"]["delta_N"] = (sim_data["MAPI"]["N"] - param_dict["MAPI"]["N0"]) if include_flags["MAPI"]['N'] else np.zeros(len(grid_x))
    param_dict["MAPI"]["delta_P"] = (sim_data["MAPI"]["P"] - param_dict["MAPI"]["P0"]) if include_flags["MAPI"]['P'] else np.zeros(len(grid_x))
    return # TODO isnt this missing something?

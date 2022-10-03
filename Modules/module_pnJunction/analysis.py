import numpy as np
import os
import tables
from scipy import integrate as intg
from io_utils import u_read
from utils import to_index, new_integrate
# from Modules.module_pnJunction.calculations import radiative_recombination
# from Modules.module_pnJunction.calculations import prep_PL
# from Modules.module_pnJunction.calculations import tau_diff
# from Modules.module_pnJunction.calculations import E_field_r
# from Modules.module_pnJunction.calculations import TTA
from Modules.module_pnJunction.calculations import CalculatedOutputs


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
                
    """
    calculated_outputs = CalculatedOutputs(data_dict["MAPI"], data_dict["Rubrene"],
                                           mapi_params, ru_params)
    data_dict["MAPI"]["E_field"] = calculated_outputs.E_field()
    data_dict["MAPI"]["delta_N"] = calculated_outputs.delta_n()
    data_dict["MAPI"]["delta_P"] = calculated_outputs.delta_p()
    data_dict["MAPI"]["RR"] = calculated_outputs.radiative_recombination()
    data_dict["MAPI"]["NRR"] = calculated_outputs.nonradiative_recombination()
            
    #### MAPI PL ####
    with tables.open_file(os.path.join(data_dirname, file_name_base + "-N.h5"), mode='r') as ifstream_N, \
        tables.open_file(os.path.join(data_dirname, file_name_base + "-P.h5"), mode='r') as ifstream_P:
        temp_N = np.array(ifstream_N.root.data)
        temp_P = np.array(ifstream_P.root.data)

    data_dict["MAPI"]["mapi_PL"] = calculated_outputs.mapi_PL(temp_N, temp_P)
    
    data_dict["MAPI"]["avg_delta_N"] = calculated_outputs.average_delta_n(temp_N)
    
    try:
        data_dict["MAPI"]["tau_diff"] = tau_diff(data_dict["MAPI"]["mapi_PL"], dt)
    except Exception:
        print("Error: failed to calculate tau_diff")
        data_dict["MAPI"]["tau_diff"] = 0
    #################
    
    if flags.get("do_sct"):
        data_dict["Rubrene"]["E_upc"] = calculated_outputs.E_field_r()
        
    # else:
    #     data_dict["Rubrene"]["E_upc"] = np.zeros_like(data_dict["Rubrene"]["T"])
    
    #### DBP PL ####
    with tables.open_file(os.path.join(data_dirname, file_name_base + "-delta_D.h5"), mode='r') as ifstream_D:
        temp_D = np.array(ifstream_D.root.data)
        
    data_dict["Rubrene"]["dbp_PL"] = calculated_outputs.dbp_PL(temp_D)
        
    ################
    
    #### TTA Rate ####
    # "Triplet-Triplet Annihilation"
    with tables.open_file(os.path.join(data_dirname, file_name_base + "-T.h5"), mode='r') as ifstream_T:
        temp_T = np.array(ifstream_T.root.data)
        
    data_dict["Rubrene"]["TTA"] = calculated_outputs.TTA(temp_T)
    ##################

    """
    for data in data_dict["__SHARED__"]:
        data_dict["__SHARED__"][data] *= convert_out[data]
        
        
    # data_dict["MAPI"]["mapi_PL"] *= mapi.iconvert_out["mapi_PL"]
    # data_dict["Rubrene"]["dbp_PL"] *= ru.iconvert_out["dbp_PL"]
    # data_dict["Rubrene"]["TTA"] *= ru.iconvert_out["TTA"]
        
    return data_dict


def submodule_prep_dataset(where_layer, layer, datatype, sim_data, params, for_integrate, 
                     i, j, nen, extra_data):
    """ Provides delta_N, delta_P, electric field,
    recombination, and spatial PL values on demand."""
    # For N, P, E-field this is just reading the data but for others we'll calculate it in situ
    layer_params = params[where_layer]
    layer_sim_data = sim_data[where_layer]
    data = None
    if (datatype in layer.s_outputs):
        data = layer_sim_data[datatype]
    
    else:
        calculated_outputs = CalculatedOutputs(sim_data["MAPI"], sim_data["Rubrene"], params["MAPI"], params["Rubrene"])
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
            
        elif (datatype == "E_field"):
            data = calculated_outputs.E_field()

        elif (datatype == "mapi_PL"):

            if for_integrate:
                rad_rec = radiative_recombination(extra_data[where_layer], layer_params)
                data = prep_PL(rad_rec, i, j, nen, layer_params, where_layer)
            else:
                rad_rec = calculated_outputs.radiative_recombination()
                data = prep_PL(rad_rec, 0, len(rad_rec)-1, False, 
                                layer_params, where_layer).flatten()
                
        elif (datatype == "E_upc"):
            data = calculated_outputs.E_field_r()
                
        elif (datatype == "dbp_PL"):

            if for_integrate:
                delta_D = extra_data[where_layer]["delta_D"]
                data = prep_PL(delta_D, i, j, nen, layer_params, where_layer)
            else:
                delta_D = layer_sim_data["delta_D"]
                data = prep_PL(delta_D, 0, len(delta_D)-1, False, 
                                layer_params, where_layer).flatten()
                
        elif (datatype == "TTA"):

            if for_integrate:
                T = extra_data[where_layer]["T"]
                data = TTA(T, i, j, nen, layer_params)
            else:
                T = layer_sim_data["T"]
                data = TTA(T, 0, len(T)-1, False, 
                            layer_params).flatten()
                
        
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

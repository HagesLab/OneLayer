import numpy as np
from io_utils import u_read
from Modules.module_Si_dualband.calculations import CalculatedOutputs, tau_diff


def submodule_get_overview_analysis(absorber_layer, params, flags, total_time,
                                    dt, tsteps, data_dirname, file_name_base):
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

    data_dict = {"Absorber": {}}

    convert_out = absorber_layer.convert_out
    iconvert_out = absorber_layer.iconvert_out
    for raw_output_name in absorber_layer.s_outputs:

        data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base,
                                             raw_output_name)
        data = []
        for tstep in tsteps:
            data.append(u_read(data_filename, t0=tstep, single_tstep=True))

        data_dict["Absorber"][raw_output_name] = np.array(data)

    calculated_outputs = CalculatedOutputs(data_dict["Absorber"], params)

    data_dict["Absorber"]["total_N"] = calculated_outputs.total_n()

    data_dict["Absorber"]["delta_N_d"] = calculated_outputs.delta_n_d()

    data_dict["Absorber"]["delta_N_ind"] = calculated_outputs.delta_n_ind()

    data_dict["Absorber"]["delta_N"] = calculated_outputs.delta_n()

    data_dict["Absorber"]["delta_P"] = calculated_outputs.delta_p()

    data_dict["Absorber"]["RR_d"] = calculated_outputs.radiative_recombination_d()

    data_dict["Absorber"]["RR_ind"] = calculated_outputs.radiative_recombination_ind()

    data_dict["Absorber"]["RR"] = calculated_outputs.radiative_recombination()

    data_dict["Absorber"]["NRR"] = calculated_outputs.nonradiative_recombination()

    data_dict["Absorber"]["trap_rate"] = calculated_outputs.trap()

    data_dict["Absorber"]["detrap_rate"] = calculated_outputs.detrap()

    # data_dict["Absorber"]["voltage"] = calculated_outputs.voltage()
    # data_dict["Absorber"]["E_field"] = calculated_outputs.E_field()

    ############################################################################
    # PL
    full_time_data = {}
    for raw_output_name in absorber_layer.s_outputs:

        data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base,
                                             raw_output_name)
        data = u_read(data_filename)

        full_time_data[raw_output_name] = np.array(data)

    full_time_outputs = CalculatedOutputs(full_time_data, params)

    data_dict["Absorber"]["PL"] = full_time_outputs.PL_integral(
        full_time_outputs.PL())
    data_dict["Absorber"]["PL_d"] = full_time_outputs.PL_integral(
        full_time_outputs.PL_d())
    data_dict["Absorber"]["PL_ind"] = full_time_outputs.PL_integral(
        full_time_outputs.PL_ind())

    data_dict["Absorber"]["avg_delta_N"] = full_time_outputs.average_delta_n()

    data_dict["Absorber"]["tau_diff"] = tau_diff(
        data_dict["Absorber"]["PL"], dt)
    ############################################################################

    for data in data_dict["Absorber"]:
        data_dict["Absorber"][data] *= convert_out[data]

    data_dict["Absorber"]["PL"] *= iconvert_out["PL"]
    data_dict["Absorber"]["PL_d"] *= iconvert_out["PL_d"]
    data_dict["Absorber"]["PL_ind"] *= iconvert_out["PL_ind"]

    return data_dict


def submodule_prep_dataset(where_layer, layer, datatype, sim_data, params, for_integrate,
                           i, j, nen, extra_data):
    """ Provides delta_N, delta_P, electric field,
    recombination, and spatial PL values on demand."""
    # For N, P this is just reading the data but for others we'll calculate it in situ
    layer_sim_data = sim_data[where_layer]
    data = None
    if (datatype in layer.s_outputs):
        data = layer_sim_data[datatype]

    else:
        calculated_outputs = CalculatedOutputs(
            sim_data[where_layer], params, i, j, nen)

        if datatype in calculated_outputs.calcs:
            data = calculated_outputs.calcs[datatype]()
        else:
            raise NotImplementedError(
                f"No calculation routine available for {datatype}")

    return data


def submodule_get_timeseries(pathname, datatype, parent_data,
                             total_time, dt, params, flags):
    """Depending on requested datatype, returns the equivalent timeseries."""

    if datatype in ["delta_N", "delta_N_d", "delta_N_ind"]:
        temp_dN = parent_data / params["Absorber"]["Total_length"]
        tdiff = tau_diff(parent_data, dt)
        return [("avg_delta_N", temp_dN),
                ("tau_diff", tdiff)]

    if datatype in ["PL", "PL_d", "PL_ind"]:
        tdiff = tau_diff(parent_data, dt)
        return [("tau_diff", tdiff)]

    else:
        return


def submodule_get_IC_regen(sim_data, param_dict, include_flags, grid_x):
    """ Set delta_N and delta_P of outgoing regenerated IC file."""
    param_dict["Absorber"]["delta_N"] = (sim_data["Absorber"]["N_d"] - param_dict["Absorber"]["N0"]
                                         ) if include_flags["Absorber"]['N_d'] else np.zeros(len(param_dict["Absorber"]['node_x']))

    # We need this variable to preserve any carriers in the indirect band
    param_dict["Absorber"]["delta_N_ind"] = (sim_data["Absorber"]["N_ind"]
                                             ) if include_flags["Absorber"]['N_ind'] else np.zeros(len(param_dict["Absorber"]['node_x']))

    param_dict["Absorber"]["delta_P"] = (sim_data["Absorber"]["P"] - param_dict["Absorber"]["P0"]
                                         ) if include_flags["Absorber"]['P'] else np.zeros(len(param_dict["Absorber"]['node_x']))

    return

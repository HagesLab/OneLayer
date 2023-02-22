import numpy as np
import os
import tables
from scipy import integrate as intg
from io_utils import u_read
from Modules.module_pnJunction.calculations import radiative_recombination
from Modules.module_pnJunction.calculations import prep_PL
from Modules.module_Si_dualband.calculations import CalculatedOutputs, tau_diff


def submodule_get_overview_analysis(any_layer, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
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

    data_dict = {"__SHARED__": {}}

    convert_out = any_layer.convert_out
    for raw_output_name in any_layer.s_outputs:

        data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base,
                                             raw_output_name)
        data = []
        for tstep in tsteps:
            data.append(u_read(data_filename, t0=tstep, single_tstep=True))

        data_dict["__SHARED__"][raw_output_name] = np.array(data)

    calculated_outputs = CalculatedOutputs(data_dict["__SHARED__"], params)

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

    data_dict["__SHARED__"]["PL"] = calculated_outputs.PL(
        temp_N, temp_P, do_integrate=True)

    data_dict["__SHARED__"]["avg_delta_N"] = calculated_outputs.average_delta_n(
        temp_N)

    # try:
    data_dict["__SHARED__"]["tau_diff"] = tau_diff(
        data_dict["__SHARED__"]["PL"], dt)
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
    layer_sim_data = sim_data[where_layer]
    data = None
    if (datatype in layer.s_outputs):
        data = layer_sim_data[datatype]

    else:
        calculated_outputs = CalculatedOutputs(sim_data[where_layer], params)
        if (datatype == "total_N"):
            data = calculated_outputs.total_n()

        elif (datatype == "delta_N"):
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
                get_these_params = ['B', 'N0', 'P0']
                these_params = calculated_outputs.get_stitched_params(
                    get_these_params)
                rad_rec = radiative_recombination(
                    extra_data[where_layer], these_params)
                data = prep_PL(rad_rec, i, j, nen)
            else:
                data = calculated_outputs.PL(
                    sim_data[where_layer]['N'], sim_data[where_layer]['P'], do_integrate=False).flatten()

        else:
            raise NotImplementedError(
                f"No calculation routine available for {datatype}")

    return data


def submodule_get_timeseries(pathname, datatype, parent_data, total_time, dt, params, flags):
    """Depending on requested datatype, returns the equivalent timeseries."""

    if datatype == "delta_N":
        temp_dN = parent_data / params["MAPI"]["Total_length"]
        return [("avg_delta_N", temp_dN)]

    if datatype == "PL":
        # try:
        tdiff = tau_diff(parent_data, dt)
        # except FloatingPointError:
        #    print("Error: failed to calculate tau_diff - effective lifetime is near infinite")
        #    tdiff = np.zeros_like(np.linspace(0, total_time, int(total_time/dt) + 1))

        return [("tau_diff", tdiff)]

    else:
        return


def submodule_get_IC_regen(sim_data, param_dict, include_flags, grid_x):
    """ Set delta_N and delta_P of outgoing regenerated IC file."""
    layer_names = ['N-type', 'buffer', 'P-type']
    l = [0] + [len(param_dict[n]['node_x']) for n in layer_names]
    l = np.cumsum(l)
    sim_data['N-type']['N'] = sim_data["__SHARED__"]['N'][l[0]:l[1]]
    sim_data['N-type']['P'] = sim_data["__SHARED__"]['P'][l[0]:l[1]]

    sim_data['buffer']['N'] = sim_data["__SHARED__"]['N'][l[1]:l[2]]
    sim_data['buffer']['P'] = sim_data["__SHARED__"]['P'][l[1]:l[2]]

    sim_data['P-type']['N'] = sim_data["__SHARED__"]['N'][l[2]:l[3]]
    sim_data['P-type']['P'] = sim_data["__SHARED__"]['P'][l[2]:l[3]]

    param_dict["N-type"]["delta_N"] = (sim_data["N-type"]["N"] - param_dict["N-type"]["N0"]
                                       ) if include_flags["__SHARED__"]['N'] else np.zeros(len(param_dict["N-type"]['node_x']))
    param_dict["N-type"]["delta_P"] = (sim_data["N-type"]["P"] - param_dict["N-type"]["P0"]
                                       ) if include_flags["__SHARED__"]['P'] else np.zeros(len(param_dict["N-type"]['node_x']))

    param_dict["buffer"]["delta_N"] = (sim_data["buffer"]["N"] - param_dict["buffer"]["N0"]
                                       ) if include_flags["__SHARED__"]['N'] else np.zeros(len(param_dict["buffer"]['node_x']))
    param_dict["buffer"]["delta_P"] = (sim_data["buffer"]["P"] - param_dict["buffer"]["P0"]
                                       ) if include_flags["__SHARED__"]['P'] else np.zeros(len(param_dict["buffer"]['node_x']))

    param_dict["P-type"]["delta_N"] = (sim_data["P-type"]["N"] - param_dict["P-type"]["N0"]
                                       ) if include_flags["__SHARED__"]['N'] else np.zeros(len(param_dict["P-type"]['node_x']))
    param_dict["P-type"]["delta_P"] = (sim_data["P-type"]["P"] - param_dict["P-type"]["P0"]
                                       ) if include_flags["__SHARED__"]['P'] else np.zeros(len(param_dict["P-type"]['node_x']))

    return

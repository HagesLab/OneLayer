
import numpy as np
from helper_structs import Parameter, Output, Layer


def define_params():
    return {
        # Mobilities of electrons (in each band) and holes
        "mu_N": Parameter(units="[cm^2 / V s]", is_edge=True,
                          valid_range=(0, np.inf)),
        "mu_N_ind": Parameter(units="[cm^2 / V s]", is_edge=True,
                              valid_range=(0, np.inf)),
        "mu_P": Parameter(units="[cm^2 / V s]", is_edge=True,
                          valid_range=(0, np.inf)),

        # Doping
        "N0": Parameter(units="[cm^-3]", is_edge=False,
                        valid_range=(0, np.inf)),
        "P0": Parameter(units="[cm^-3]", is_edge=False,
                        valid_range=(0, np.inf)),

        # Trapping and band exchange
        "tau_C": Parameter(units="[ns]", is_edge=False,
                           valid_range=(0, np.inf)),
        "tau_E": Parameter(units="[ns]", is_edge=False,
                           valid_range=(0, np.inf)),

        # Recombination
        "B": Parameter(units="[cm^3 / s]", is_edge=False,
                       valid_range=(0, np.inf)),
        "B_ind": Parameter(units="[cm^3 / s]", is_edge=False,
                           valid_range=(0, np.inf)),
        "tau_N": Parameter(units="[ns]", is_edge=False,
                           valid_range=(0, np.inf)),
        "tau_P": Parameter(units="[ns]", is_edge=False,
                           valid_range=(0, np.inf)),
        "Sf": Parameter(units="[cm / s]", is_edge=False,
                        is_space_dependent=False, valid_range=(0, np.inf)),
        "Sb": Parameter(units="[cm / s]", is_edge=False,
                        is_space_dependent=False, valid_range=(0, np.inf)),
        "CN": Parameter(units="[cm^6 / s]", is_edge=False,
                        valid_range=(0, np.inf)),
        "CP": Parameter(units="[cm^6 / s]", is_edge=False,
                        valid_range=(0, np.inf)),

        # Temperature
        "Temp": Parameter(units="[K]", is_edge=True, valid_range=(0, np.inf)),

        # Electric field stuff
        "eps_perm": Parameter(units="", is_edge=False,
                              valid_range=(0, np.inf)),
        "delta_N": Parameter(units="[cm^-3]", is_edge=False,
                             valid_range=(0, np.inf)),
        "delta_P": Parameter(units="[cm^-3]", is_edge=False,
                             valid_range=(0, np.inf)),
        "Eg": Parameter(units="[eV]", is_edge=False),
        "n_affin": Parameter(units="[eV]", is_edge=False)
    }


def define_simulation_outputs(layer):
    return {
        # Electron (in each band) and hole profiles
        "N_d": Output("N", units="[cm^-3]", integrated_units="[cm^-2]",
                      xlabel="nm", xvar="position", is_edge=False,
                      layer=layer, yscale='symlog', yfactors=(1e-4, 1e1)),
        "N_ind": Output("N_ind", units="[cm^-3]", integrated_units="[cm^-2]",
                        xlabel="nm", xvar="position", is_edge=False,
                        layer=layer, yscale='symlog', yfactors=(1e-4, 1e1)),
        "P": Output("P", units="[cm^-3]", integrated_units="[cm^-2]",
                    xlabel="nm", xvar="position", is_edge=False,
                    layer=layer, yscale='symlog', yfactors=(1e-4, 1e1)),
    }


def define_calculated_outputs(layer):
    return {
        "voltage": Output("Voltage", units="[V]", integrated_units="[V nm]",
                          xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "E_field": Output("Electric Field", units="[V/nm]", integrated_units="[V]",
                          xlabel="nm", xvar="position", is_edge=True, layer=layer),
        "total_N": Output("Total_N", units="[cm^-3]", integrated_units="[cm^-2]",
                          xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "delta_N": Output("delta_N", units="[cm^-3]", integrated_units="[cm^-2]",
                          xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "delta_N_d": Output("delta_N (direct)", units="[cm^-3]",
                            integrated_units="[cm^-2]", xlabel="nm",
                            xvar="position", is_edge=False, layer=layer),
        "delta_N_ind": Output("delta_N (indirect)", units="[cm^-3]",
                              integrated_units="[cm^-2]", xlabel="nm",
                              xvar="position", is_edge=False, layer=layer),
        "delta_P": Output("delta_P", units="[cm^-3]", integrated_units="[cm^-2]",
                          xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "RR": Output("Radiative Rec.", units="[cm^-3 s^-1]",
                     integrated_units="[cm^-3 s^-1]", xlabel="nm", xvar="position",
                     is_edge=False, layer=layer),
        "NRR": Output("SRH Rec.", units="[cm^-3 s^-1]",
                      integrated_units="[cm^-3 s^-1]", xlabel="nm", xvar="position",
                      is_edge=False, layer=layer),
        "PL": Output("TRPL", units="[cm^-3 s^-1]", integrated_units="[cm^-2 s^-1]",
                     xlabel="ns", xvar="time", is_edge=False, layer=layer),
        "PL_d": Output("TRPL (direct)", units="[cm^-3 s^-1]",
                       integrated_units="[cm^-2 s^-1]",
                       xlabel="ns", xvar="time", is_edge=False, layer=layer),
        "PL_ind": Output("TRPL (indirect)", units="[cm^-3 s^-1]",
                         integrated_units="[cm^-2 s^-1]",
                         xlabel="ns", xvar="time", is_edge=False, layer=layer),
        "tau_diff": Output("tau_diff", units="[ns]", xlabel="ns", xvar="time",
                           is_edge=False, layer=layer, analysis_plotable=False),
        "avg_delta_N": Output("avg_delta_N", units="[cm^-3]", xlabel="ns",
                              xvar="time", is_edge=False, layer=layer,
                              analysis_plotable=False),

    }


def define_convert_in():
    convert_in = {
        "mu_N": ((1e7) ** 2) / (1e9),
        "mu_N_ind": ((1e7) ** 2) / (1e9),
        "mu_P": ((1e7) ** 2) / (1e9),  # [cm^2 / V s] to [nm^2 / V ns]
        "N0": ((1e-7) ** 3),
        "P0": ((1e-7) ** 3),           # [cm^-3] to [nm^-3]
        "B_ind": ((1e7) ** 3) / (1e9),
        "B": ((1e7) ** 3) / (1e9),     # [cm^3 / s] to [nm^3 / ns]
        "tau_C": 1,
        "tau_E": 1,
        "tau_N": 1,
        "tau_P": 1,                    # [ns]
        "Sf": (1e7) / (1e9),
        "Sb": (1e7) / (1e9),           # [cm / s] to [nm / ns]
        "CN": 1e33, "CP": 1e33,        # [cm^6 / s] to [nm^6 / ns]
        "Temp": 1,
        "eps_perm": 1,
        "delta_N": ((1e-7) ** 3),
        "delta_P": ((1e-7) ** 3),
        "Eg": 1, "n_affin": 1,
        "avg_delta_N": ((1e-7) ** 3),
        "N_d": ((1e-7) ** 3),
        "N_ind": ((1e-7) ** 3),
        "total_N": ((1e-7) ** 3),
        "delta_N_d": ((1e-7) ** 3),
        "delta_N_ind": ((1e-7) ** 3),
        "P": ((1e-7) ** 3),            # [cm^-3] to [nm^-3]
        "voltage": 1,
        "E_field": 1,
        "tau_diff": 1,
    }

    # These really exist only for the convert_out - so outputs are
    # displayed in cm and s instead of nm and ns
    convert_in["RR"] = convert_in["B"] * (convert_in["N_d"] * convert_in["P"])
    convert_in["NRR"] = convert_in["N_d"] * 1e-9
    convert_in["PL"] = convert_in["RR"]
    convert_in["PL_d"] = convert_in["PL"]
    convert_in["PL_ind"] = convert_in["PL"]

    return convert_in


def define_iconvert_in():
    return {
        "N_d": 1e7,
        "N_ind": 1e7,
        "total_N": 1e7,
        "P": 1e7,
        "delta_N": 1e7,
        "delta_N_d": 1e7,
        "delta_N_ind": 1e7,
        "delta_P": 1e7,
        "voltage": 1,
        "E_field": 1,
        "RR": 1e7,
        "NRR": 1e7,
        "PL": 1e7,
        "PL_d": 1e7,
        "PL_ind": 1e7
    }


def define_layers():

    # Lists of conversions into and out of TEDs units (e.g. nm/s)
    # from common units (e.g. cm/s)
    # Multiply the parameter values the user enters in common units
    # by the corresponding coefficient in this dictionary to convert into TEDs units
    convert_in = define_convert_in()
    iconvert_in = define_iconvert_in()

    # we can now initialize the 2 layers with the previously defined components
    layers = {
        "Absorber": Layer(
                define_params(),
                define_simulation_outputs("Absorber"),
                define_calculated_outputs("Absorber"),
                "[nm]",
                convert_in,
                iconvert_in
            ),
    }

    return layers


def define_flags():
    return {
        "check_do_ss": ("Steady State Input", 1, 0),
    }

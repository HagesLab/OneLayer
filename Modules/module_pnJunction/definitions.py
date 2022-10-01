
import numpy as np
from helper_structs import Parameter, Output, Layer

# All three layers have the same parameters
def define_ntype_params():
    return {
        "mu_N":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
        "mu_P":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
        "N0":Parameter(units="[cm^-3]", is_edge=False, valid_range=(0,np.inf)), 
        "P0":Parameter(units="[cm^-3]", is_edge=False, valid_range=(0,np.inf)), 
        "B":Parameter(units="[cm^3 / s]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_N":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_P":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "Sf":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "Sb":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "Cn":Parameter(units="[cm^6 / s]", is_edge=False, valid_range=(0,np.inf)),
        "Cp":Parameter(units="[cm^6 / s]", is_edge=False, valid_range=(0,np.inf)),
        "temperature":Parameter(units="[K]", is_edge=True, valid_range=(0,np.inf)), 
        "rel_permitivity":Parameter(units="", is_edge=False, valid_range=(0,np.inf)), 
        "delta_N":Parameter(units="[cm^-3]", is_edge=False, valid_range=(0,np.inf)), 
        "delta_P":Parameter(units="[cm^-3]", is_edge=False, valid_range=(0,np.inf)), 
        "Eg":Parameter(units="[eV]", is_edge=False), 
        "electron_affinity":Parameter(units="[eV]", is_edge=False)
    }


def define_ntype_simulation_outputs(layer):
    return {
        "N":Output("N", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer=layer, yscale='symlog', yfactors=(1e-4,1e1)), 
        "P":Output("P", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position",is_edge=False, layer=layer, yscale='symlog', yfactors=(1e-4,1e1)),
    }

def define_ntype_calculated_outputs(layer):
    return {
        "E_field":Output("Electric Field", units="[V/nm]", integrated_units="[V]", xlabel="nm", xvar="position",is_edge=True, layer=layer),
        "delta_N":Output("delta_N", units="[cm^-3]", integrated_units="[cm^-2]", xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "delta_P":Output("delta_P", units="[cm^-3]", integrated_units="[cm^-2]", xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "RR":Output("Radiative Recombination", units="[cm^-3 s^-1]", integrated_units="[cm^-3 s^-1]", xlabel="nm", xvar="position",is_edge=False, layer=layer),
        "NRR":Output("Non-radiative Recombination", units="[cm^-3 s^-1]", integrated_units="[cm^-3 s^-1]", xlabel="nm", xvar="position", is_edge=False, layer=layer),
        "PL":Output("TRPL", units="[cm^-3 s^-1]", integrated_units="[cm^-2 s^-1]", xlabel="ns", xvar="time", is_edge=False, layer=layer),
        "tau_diff":Output("tau_diff", units="[ns]", xlabel="ns", xvar="time", is_edge=False, layer=layer, analysis_plotable=False),
        "avg_delta_N":Output("avg_delta_N", units="[cm^-3]", xlabel="ns", xvar="time", is_edge=False, layer=layer, analysis_plotable=False),
        
    }

def define_ntype_convert_in():
    ntype_convert_in = {
        "mu_N": ((1e7) ** 2) / (1e9),
        "mu_P": ((1e7) ** 2) / (1e9), # [cm^2 / V s] to [nm^2 / V ns]
        "N0": ((1e-7) ** 3),
        "P0": ((1e-7) ** 3),          # [cm^-3] to [nm^-3]
        "B": ((1e7) ** 3) / (1e9),    # [cm^3 / s] to [nm^3 / ns]
        "tau_N": 1,
        "tau_P": 1,                   # [ns]
        "Sf": (1e7) / (1e9),
        "Sb": (1e7) / (1e9),          # [cm / s] to [nm / ns]
        "Cn": 1e33, "Cp": 1e33,       # [cm^6 / s] to [nm^6 / ns]
        "temperature": 1,
        "rel_permitivity":1,
        "delta_N": ((1e-7) ** 3),
        "delta_P": ((1e-7) ** 3),
        "Eg": 1, "electron_affinity": 1,
        "avg_delta_N": ((1e-7) ** 3),
        "N": ((1e-7) ** 3),
        "P": ((1e-7) ** 3),           # [cm^-3] to [nm^-3]
        "E_field": 1,
        "tau_diff": 1,
    }

    # These really exist only for the convert_out - so outputs are displayed in cm and s instead of nm and ns
    ntype_convert_in["RR"] = ntype_convert_in["B"] * ntype_convert_in["N"] * ntype_convert_in["P"] # [cm^-3 s^-1] to [m^-3 ns^-1]
    ntype_convert_in["NRR"] = ntype_convert_in["N"] * 1e-9 # [cm^-3 s^-1] to [nm^-3 ns^-1]
    ntype_convert_in["PL"] = ntype_convert_in["RR"]

    return ntype_convert_in

def define_ntype_iconvert_in():
    return {
        "N": 1e7,
        "P": 1e7,
        "delta_N": 1e7,
        "delta_P": 1e7, # cm to nm
        "E_field": 1, # nm to nm
        "RR": 1e7,
        "NRR": 1e7,
        "PL": 1e7
    }

def define_layers():
    
    # Lists of conversions into and out of TEDs units (e.g. nm/s)
    # from common units (e.g. cm/s)
    # Multiply the parameter values the user enters in common units
    # by the corresponding coefficient in this dictionary to convert into TEDs units
    convert_in = define_ntype_convert_in()
    iconvert_in = define_ntype_iconvert_in()

    # we can now initialize the 2 layers with the previously defined components
    layers = {
        "N-type": Layer(
                define_ntype_params(),
                define_ntype_simulation_outputs("N-type"),
                define_ntype_calculated_outputs("N-type"),
                "[nm]",
                convert_in,
                iconvert_in
            ),
        "buffer": Layer(
                define_ntype_params(),
                define_ntype_simulation_outputs("buffer"),
                define_ntype_calculated_outputs("buffer"),
                "[nm]",
                convert_in,
                iconvert_in
            ),
        "P-type": Layer(
                define_ntype_params(),
                define_ntype_simulation_outputs("P-type"),
                define_ntype_calculated_outputs("P-type"),
                "[nm]",
                convert_in,
                iconvert_in
            )
    }

    return layers


def define_flags():
    return {
        "check_do_ss": ("Steady State Input",1, 0),
    }
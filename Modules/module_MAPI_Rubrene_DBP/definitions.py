
import numpy as np
from helper_structs import Parameter, Output, Layer

def define_mapi_params():
    return {
        "mu_N":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
        "mu_P":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
        "N0":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
        "P0":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
        "B":Parameter(units="[cm^3 / s]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_N":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_P":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "Sf":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "Sb":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "Cn":Parameter(units="[cm^6 / s]", is_edge=False, valid_range=(0,np.inf)),
        "Cp":Parameter(units="[cm^6 / s]", is_edge=False, valid_range=(0,np.inf)),
        "MAPI_temperature":Parameter(units="[K]", is_edge=True, valid_range=(0,np.inf)), 
        "rel_permitivity":Parameter(units="", is_edge=True, valid_range=(0,np.inf)), 
        "delta_N":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
        "delta_P":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
    }


def define_rubrene_params():
    return {
        "mu_P_up":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)),
        "mu_T":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
        "mu_S":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)),
        "T0":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_T":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_S":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "tau_D":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
        "k_fusion":Parameter(units="[cm^3 / s]", is_edge=False, valid_range=(0,np.inf)),
        "k_0":Parameter(units="[nm^3 s^-1]", is_edge=False, valid_range=(0,np.inf)),
        "Ssct":Parameter(units="[cm^3 / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "Sp":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "St":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
        "W_VB":Parameter(units="[eV]", is_edge=False, is_space_dependent=False), 
        "Rubrene_temperature":Parameter(units="[K]", is_edge=True, valid_range=(0,np.inf)), 
        "uc_permitivity":Parameter(units="", is_edge=True, valid_range=(0,np.inf)), 
        "delta_T":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
        "delta_S":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
        "delta_D":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
    }


def define_mapi_simulation_outputs():
    return {
        "N":Output("N", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="MAPI", yscale='symlog', yfactors=(1e-4,1e1)), 
        "P":Output("P", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position",is_edge=False, layer="MAPI", yscale='symlog', yfactors=(1e-4,1e1)),
    }


def define_rubrene_simulation_outputs():
    return {
        "P_up":Output("P_up", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position",is_edge=False, layer="Rubrene", yscale='symlog', yfactors=(1e-4,1e1)),
        "T":Output("T", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="Rubrene", yscale='symlog', yfactors=(1e-4,1e1)), 
        "delta_S":Output("delta_S", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position",is_edge=False, layer="Rubrene", yscale='symlog', yfactors=(1e-4,1e1)),
        "delta_D":Output("delta_D", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position",is_edge=False, layer="Rubrene", yscale='symlog', yfactors=(1e-4,1e1)),
    }


def define_mapi_calculated_outputs():
    return {
        "E_field":Output("Electric Field", units="[V/nm]", integrated_units="[V]", xlabel="nm", xvar="position",is_edge=True, layer="MAPI"),
        "delta_N":Output("delta_N", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="MAPI"),
        "delta_P":Output("delta_P", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="MAPI"),
        "RR":Output("Radiative Recombination", units="[carr / cm^3 s]", integrated_units="[carr / cm^2 s]", xlabel="nm", xvar="position",is_edge=False, layer="MAPI"),
        "NRR":Output("Non-radiative Recombination", units="[carr / cm^3 s]", integrated_units="[carr / cm^2 s]", xlabel="nm", xvar="position", is_edge=False, layer="MAPI"),
        "mapi_PL":Output("MAPI TRPL", units="[phot / cm^3 s]", integrated_units="[phot / cm^2 s]", xlabel="ns", xvar="time", is_edge=False, layer="MAPI"),
        "tau_diff":Output("tau_diff", units="[ns]", xlabel="ns", xvar="time", is_edge=False, layer="MAPI", analysis_plotable=False),
        "avg_delta_N":Output("avg_delta_N", units="[carr / cm^3]", xlabel="ns", xvar="time", is_edge=False, layer="MAPI", analysis_plotable=False),
        "eta_MAPI":Output("MAPI eff.", units="", xlabel="ns", xvar="time", is_edge=False, layer="MAPI", analysis_plotable=False)
    }


def define_rubrene_calculated_outputs():
    return {
        "E_upc":Output("E (Upc)", units="[V/nm]", integrated_units="[V]", xlabel="nm", xvar="position",is_edge=True, layer="Rubrene"),
        "dbp_PL":Output("DBP TRPL", units="[phot / cm^3 s]", integrated_units="[phot / cm^2 s]", xlabel="ns", xvar="time", is_edge=False, layer="Rubrene"),
        "TTA":Output("TTA Rate", units="[phot / cm^3 s]", integrated_units="[phot / cm^2 s]", xlabel="ns", xvar="time", is_edge=False, layer="Rubrene"),
        "T_form_eff":Output("Triplet form. eff.", units="", xlabel="ns", xvar="time", is_edge=False, layer="Rubrene", analysis_plotable=False),
        "T_anni_eff":Output("Triplet anni. eff.", units="", xlabel="ns", xvar="time", is_edge=False, layer="Rubrene", analysis_plotable=False),
        "S_up_eff":Output("Singlet upc. eff.", units="", xlabel="ns", xvar="time", is_edge=False, layer="Rubrene", analysis_plotable=False),
        "eta_UC":Output("DBP. eff.", units="", xlabel="ns", xvar="time", is_edge=False, layer="Rubrene", analysis_plotable=False)
    }


def define_mapi_convert_in():
    mapi_convert_in = {
        "mu_N": ((1e7) ** 2) / (1e9),
        "mu_P": ((1e7) ** 2) / (1e9), # [cm^2 / V s] to [nm^2 / V ns]
        "N0": ((1e-7) ** 3),
        "P0": ((1e-7) ** 3),                   # [cm^-3] to [nm^-3]
        "B": ((1e7) ** 3) / (1e9),                                  # [cm^3 / s] to [nm^3 / ns]
        "tau_N": 1,
        "tau_P": 1,                                     # [ns]
        "Sf": (1e7) / (1e9),
        "Sb": (1e7) / (1e9),                   # [cm / s] to [nm / ns]
        "Cn": 1e33, "Cp": 1e33,                                   # [cm^6 / s] to [nm^6 / ns]
        "MAPI_temperature": 1,
        "rel_permitivity":1,
        "delta_N": ((1e-7) ** 3),
        "delta_P": ((1e-7) ** 3),
        "avg_delta_N": ((1e-7) ** 3),
        "N": ((1e-7) ** 3),
        "P": ((1e-7) ** 3),                     # [cm^-3] to [nm^-3]
        "E_field": 1,
        "tau_diff": 1,
        "eta_MAPI":1
    }

    # These really exist only for the convert_out - so outputs are displayed in cm and s instead of nm and ns
    mapi_convert_in["RR"] = mapi_convert_in["B"] * mapi_convert_in["N"] * mapi_convert_in["P"] # [cm^-3 s^-1] to [m^-3 ns^-1]
    mapi_convert_in["NRR"] = mapi_convert_in["N"] * 1e-9 # [cm^-3 s^-1] to [nm^-3 ns^-1]
    mapi_convert_in["mapi_PL"] = mapi_convert_in["RR"]

    return mapi_convert_in


def define_rubrene_convert_in():
    rubrene_convert_in = {
        "mu_P_up":1e5,
        "mu_T": 1e5, "mu_S": 1e5,                         # [cm^2 / V s] to [nm^2 / V ns]
        "T0": 1e-21,                                      # [cm^-3] to [nm^-3]
        "tau_T": 1, "tau_S": 1, "tau_D": 1,               # [ns]
        "k_fusion":1e12,                                  # [cm^3 / s] to [nm^3 / ns]
        "k_0":1e-9,                                       # [nm^3 / s] to [nm^3 / ns]
        "Rubrene_temperature":1, "uc_permitivity":1,
        "Ssct":1e12,
        "St": (1e7) / (1e9),                              # [cm/s] to [nm/ns]
        "Sp":1e-2,
        "W_VB":1,
        "P_up":1e-21,
        "delta_T":1e-21, "delta_S":1e-21, "delta_D":1e-21,# [cm^-3] to [nm^-3]
        "T":1e-21, "S":1e-21, "D":1e-21,                   # [cm^-3] to [nm^-3]
        "E_upc":1,
        "T_form_eff":1, "T_anni_eff":1, "S_up_eff":1,
        "eta_UC":1
    }
        
    rubrene_convert_in["dbp_PL"] = rubrene_convert_in["delta_D"] * 1e-9 # [cm^-3 s^-1] to [nm^-3 ns^-1]
    rubrene_convert_in["TTA"] = rubrene_convert_in["k_fusion"] * rubrene_convert_in["delta_T"] ** 2

    return rubrene_convert_in


def define_mapi_iconvert_in():
    return {
        "N": 1e7,
        "P": 1e7,
        "delta_N": 1e7,
        "delta_P": 1e7, # cm to nm
        "E_field": 1, # nm to nm
        "RR": 1e7,
        "NRR": 1e7,
        "mapi_PL": 1e7
    }


def define_rubrene_iconvert_in():
    return {
        "P_up": 1e7,
        "T": 1e7,
        "delta_S": 1e7,
        "delta_D": 1e7,
        "dbp_PL": 1e7,
        "TTA": 1e7,
        "E_upc":1
    }


def define_layers():
    mapi_params = define_mapi_params()
    rubrene_params = define_rubrene_params()

    # List of all variables active during the finite difference simulating        
    # calc_inits() must return values for each of these or an error will be raised!
    mapi_simulation_outputs = define_mapi_simulation_outputs()
    rubrene_simulation_outputs = define_rubrene_simulation_outputs()
    
    # List of all variables calculated from those in simulation_outputs_dict
    mapi_calculated_outputs = define_mapi_calculated_outputs()
    rubrene_calculated_outputs = define_rubrene_calculated_outputs()

    # Lists of conversions into and out of TEDs units (e.g. nm/s)
    # from common units (e.g. cm/s)
    # Multiply the parameter values the user enters in common units
    # by the corresponding coefficient in this dictionary to convert into TEDs units
    mapi_convert_in = define_mapi_convert_in()
    mapi_iconvert_in = define_mapi_iconvert_in()
    rubrene_convert_in = define_rubrene_convert_in()
    ru_iconvert_in = define_rubrene_iconvert_in()

    # we can now initialize the 2 layers with the previously defined components
    layers = {
        "MAPI": Layer(
                mapi_params,
                mapi_simulation_outputs,
                mapi_calculated_outputs,
                "[nm]",
                mapi_convert_in,
                mapi_iconvert_in
            ),
        "Rubrene": Layer(
                rubrene_params,
                rubrene_simulation_outputs,
                rubrene_calculated_outputs,
                "[nm]",
                rubrene_convert_in,
                ru_iconvert_in
            )
    }

    return layers


def define_flags():
    return {
        "do_fret": ("Include Fret",1, 0),
        "check_do_ss": ("Steady State Input",1, 0),
        "no_upconverter": ("Deactivate Upconverter", 1, 0),
        "predict_sst": ("Predict S.S. Triplet Density", 1, 0),
        "do_sct": ("Sequential Charge Transfer", 1, 0)
    }

import numpy as np
from utils import to_array


class MAPI_Rubrene_Initial_Conditions():
    def __init__(self, mapi, rubrene):
        """Taking information from MAPI and Rubrene 'Layer' types
        we generate and return initial conditions in a dictionary."""

        self.mapi = mapi
        self.rubrene = rubrene
        
        self.calc_inits_from_layers()
        
        self.uniformize_inits_to_arrays()


    def calc_inits_from_layers(self):
        """ Physical calculations of initial distributions."""

        self.init_N = (
            self.mapi.params["N0"].value + self.mapi.params["delta_N"].value
            ) * self.mapi.convert_in["N"]

        self.init_P = (
            self.mapi.params["P0"].value + self.mapi.params["delta_P"].value
            ) * self.mapi.convert_in["P"]

        self.init_T = (
            self.rubrene.params["T0"].value + self.rubrene.params["delta_T"].value
            ) * self.rubrene.convert_in["T"]

        self.init_S = self.rubrene.params["delta_S"].value * self.rubrene.convert_in["S"]

        self.init_D = self.rubrene.params["delta_D"].value * self.rubrene.convert_in["D"]

        self.init_P_up = 0


    

    def uniformize_inits_to_arrays(self):
        """ 'Typecast' single values to uniform arrays. """

        def generate_array_if_not_yet(array_to_check, array_length):
            if not isinstance(array_to_check, np.ndarray):
                array_to_check = to_array(array_to_check, array_length, False)

            return array_to_check

        mapi_length = len(self.mapi.grid_x_nodes)
        rubrene_length = len(self.rubrene.grid_x_nodes)

        self.init_N = generate_array_if_not_yet(self.init_N, mapi_length)
        self.init_P = generate_array_if_not_yet(self.init_P, mapi_length)
        self.init_T = generate_array_if_not_yet(self.init_T, rubrene_length)
        self.init_S = generate_array_if_not_yet(self.init_S, rubrene_length)
        self.init_D = generate_array_if_not_yet(self.init_D, rubrene_length)
        self.init_P_up = generate_array_if_not_yet(self.init_P_up, rubrene_length)


    def format_inits_to_dict(self):
        """Returning the calculated initial arrays
        organized in a dictionary as per contract."""
        return {
            "N": self.init_N,
            "P": self.init_P,
            "T": self.init_T,
            "delta_S": self.init_S,
            "delta_D": self.init_D,
            "P_up": self.init_P_up
        }

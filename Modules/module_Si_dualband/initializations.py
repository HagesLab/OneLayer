import numpy as np
from utils import to_array


class Si_dualband_Initial_Conditions():
    def __init__(self, absorber):
        """Taking information from the absorber 'Layer'
        we generate and return initial conditions in a dictionary."""

        self.absorber = absorber

        self.calc_inits_from_layers()

        self.uniformize_inits_to_arrays()

    def calc_inits_from_layers(self):
        """ Physical calculations of initial distributions.
            Electrons in direct band, electrons in indirect band,
            and holes in valence band
            We may want to calculate a nonzero indirect absorption
            at some point
        """

        self.init_N_d = (
            self.absorber.params["N0"].value +
            self.absorber.params["delta_N"].value
        ) * self.absorber.convert_in["N_d"]

        self.init_N_ind = self.absorber.params["delta_N_ind"].value * \
            self.absorber.convert_in["N_ind"]

        self.init_P = (
            self.absorber.params["P0"].value +
            self.absorber.params["delta_P"].value
        ) * self.absorber.convert_in["P"]

    def uniformize_inits_to_arrays(self):
        """ 'Typecast' single values to uniform arrays. """

        def generate_array_if_not_yet(array_to_check, array_length):
            if not isinstance(array_to_check, np.ndarray):
                array_to_check = to_array(array_to_check, array_length, False)

            return array_to_check

        nx = len(self.absorber.grid_x_nodes)

        self.init_N_d = generate_array_if_not_yet(self.init_N_d, nx)
        self.init_N_ind = generate_array_if_not_yet(self.init_N_ind, nx)
        self.init_P = generate_array_if_not_yet(self.init_P, nx)

    def format_inits_to_dict(self):
        """Returning the calculated initial arrays
        organized in a dictionary as required by
        OneD_Model template."""
        return {
            "N_d": self.init_N_d,
            "N_ind": self.init_N_ind,
            "P": self.init_P
        }

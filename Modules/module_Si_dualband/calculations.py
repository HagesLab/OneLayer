import numpy as np
from scipy.linalg import solve_banded
from scipy import integrate as intg
from utils import generate_shared_x_array, to_array, to_index, new_integrate

q = 1.0                     # [e]
q_C = 1.602e-19             # [C]
kB = 8.61773e-5             # [eV / K]
eps0 = 8.854e-12 * 1e-9     # [C/V-m] to [C/V-nm]


def V_poisson_2D(dx, N, P, n0, p0, eps, V0=0, VL=0):
    if N.ndim == 2:
        num_xsteps = len(N[0])
        num_tsteps = len(N)
        V = np.zeros((num_tsteps, num_xsteps))
        for i in range(len(V)):  # For each timestep
            V[i] = V_poisson(dx, N[i], P[i], n0, p0, eps, V0, VL, mode='ocv')
        return V
    elif N.ndim == 1:
        num_xsteps = len(N)
        V = V_poisson(dx, N, P, n0, p0, eps, V0, VL, mode='ocv')
        return V


def V_poisson(dx, N, P, n0, p0, eps, k0=0, kL=0, mode='ocv'):
    if N.ndim == 2:
        raise NotImplementedError("Only works with one time step at a time")
    elif N.ndim == 1:
        num_xsteps = len(N)

    alpha = -(q_C * dx**2) / (eps * eps0)  # [V nm^3]
    rhs = np.zeros(num_xsteps)
    rhs[:] = alpha * ((P - p0) - (N - n0))

    coef_matrix = np.zeros((3, num_xsteps))

    coef_matrix[0, 2:] = 1
    coef_matrix[1, 1:-1] = -2
    coef_matrix[2, :-2] = 1

    if mode == 'fv':
        raise NotImplementedError("No fixed voltage BC yet")
        # Fixed voltage #####
        # These set the boundary equations V[0] = V0, V[L] = VL
        # coef_matrix[1,0] = 1
        # coef_matrix[0,1] = 0
        # coef_matrix[1,-1] = 1
        # coef_matrix[2,-2] = 0

        # # These set the actual values of V0 and VL
        # rhs[0] = p.V0 # [V]
        # rhs[-1] = p.VL
        ####################
    elif mode == 'ocv':
        # Open circuit
        # Where k0 is the fixed voltage [V] at left bound and kL is the fixed
        # Efield [V/nm] at right bound
        coef_matrix[1, 0] = -3
        coef_matrix[0, 1] = 1
        coef_matrix[1, -1] = -1
        coef_matrix[2, -2] = 1
        # These set the values of k0 and kL
        rhs[0] = rhs[0] - 2 * k0
        rhs[-1] = rhs[-1] + kL * dx[-1]

    return solve_banded((1, 1), coef_matrix, rhs)


def E_field_from_V(V, dx):
    if V.ndim == 2:
        E = -np.diff(V, axis=1) / dx
        E = np.concatenate([E[:, :1], E, E[:, -1:]], axis=1)
    elif V.ndim == 1:
        E = -np.diff(V) / dx
        E = np.hstack([E[0], E, E[-1]])
    return E


def delta_n(N, n0):
    """Calculate above-equilibrium electron density from N, n0"""
    return N - n0


def delta_p(P, p0):
    """Calculate above-equilibrium hole density from P, p0"""
    return P - p0


def radiative_recombination(N, P, B, n0, p0):
    """Calculate radiative recombination
        N - electrons, P - holes
    """
    return B * (N * P - n0 * p0)


def nonradiative_recombination(N_d, P, n0, p0, tau_N, tau_P):
    """Calculate nonradiative recombination using SRH model
        Assumes quasi steady state trap level occupation
        N_d - electrons in direct band, P - holes
    """
    return (N_d * P - n0 * p0) / ((tau_N * P) + (tau_P * N_d))


def trap(N_d, n0, tau_C):
    """Calculate trapping rate
        N_d - electrons in direct band
    """
    return (N_d - n0) / tau_C


def detrap(N_ind, tau_E):
    """Calculate trapping rate
        N_d - electrons in direct band
    """
    return (N_ind) / tau_E


def tau_diff(y, dt):
    """
    Calculates differential lifetime

    Parameters
    ----------
    y : 1D ndarray
        Time-resolved PL or delta_N array.
    dt : float
        Time step size.

    Returns
    -------
    1D ndarray
        tau_diff.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        ln_y = np.where(y <= 0, 0, np.log(y))

    dln_y = np.zeros(len(ln_y))
    dln_y[0] = (ln_y[1] - ln_y[0]) / dt
    dln_y[-1] = (ln_y[-1] - ln_y[-2]) / dt
    dln_y[1:-1] = (np.roll(ln_y, -1)[1:-1] -
                   np.roll(ln_y, 1)[1:-1]) / (2*dt)

    with np.errstate(invalid='ignore', divide='ignore'):
        dln_PLdt = np.where(dln_y == 0, 0, -(dln_y ** -1))
    return dln_PLdt


def prep_PL(rad_rec, i, j, need_extra_node):
    """
    Calculates PL(x,t) given radiative recombination data plus propogation contributions.

    Parameters
    ----------
    radRec : 1D or 2D ndarray
        Radiative Recombination(x,t) values.
    i : int
        Leftmost node index to calculate for.
    j : int
        Rightmost node index to calculate for.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered.
        Most slices involving the index j should include j+1 too
    Returns
    -------
    PL_base : 2D ndarray
        PL(t,x)

    """

    if rad_rec.ndim == 2:  # for integrals of partial thickness
        if need_extra_node:
            rad_rec = rad_rec[:, i:j+2]
        else:
            rad_rec = rad_rec[:, i:j+1]

    PL_base = (rad_rec)

    return PL_base


class CalculatedOutputs():

    def __init__(self, sim_outputs, params, left, right, has_extra_node):
        self.sim_outputs = sim_outputs

        self.layer_names = ["Absorber"]
        self.total_lengths = [params[layer_name]["Total_length"]
                              for layer_name in self.layer_names]

        self.grid_x_edges = [params[layer_name]['edge_x']
                             for layer_name in self.layer_names]
        self.grid_x_nodes = [params[layer_name]['node_x']
                             for layer_name in self.layer_names]
        self.dx = np.diff(generate_shared_x_array(
            True, self.grid_x_edges, self.total_lengths))
        self.inter_dx = ((self.dx + np.roll(self.dx, -1))/2)[:-1]

        self.params = params

        # Truncate param arrays to match i and j set by new_integrate
        self.left = left
        self.right = right
        if has_extra_node:
            self.right += 1

        self.calcs = {"total_N": self.total_n,
                      "delta_N_d": self.delta_n_d,
                      "delta_N_ind": self.delta_n_ind,
                      "delta_N": self.delta_n,
                      "delta_P": self.delta_p,
                      "RR_d":  self.radiative_recombination_d,
                      "RR_ind": self.radiative_recombination_ind,
                      "RR": self.radiative_recombination,
                      "NRR": self.nonradiative_recombination,
                      # "voltage": self.voltage,
                      # "E_field": self.E_field,
                      "PL_d": self.PL_d,
                      "PL_ind": self.PL_ind,
                      "PL": self.PL,
                      "trap_rate": self.trap,
                      "detrap_rate": self.detrap}

        return

    def get_stitched_params(self, get_these_params):
        these_params = {}

        for p in get_these_params:
            these_params[p] = [to_array(self.params[layer_name][p], len(self.grid_x_nodes[i]),
                                        is_edge=False)
                               for i, layer_name in enumerate(self.layer_names)]
            these_params[p] = generate_shared_x_array(
                False, these_params[p], total_lengths=None)
            these_params[p] = these_params[p][self.left:self.right+1]

        return these_params

    def total_n(self):
        return self.sim_outputs['N_d'] + self.sim_outputs['N_ind']

    # def voltage(self):
    #     """Calculate voltage from N, P"""
    #     get_these_params = ['N0', 'P0', 'rel_permitivity']
    #     these_params = self.get_stitched_params(get_these_params)

    #     return V_poisson_2D(self.dx, self.sim_outputs['N'], self.sim_outputs['P'],
    #                         these_params['N0'], these_params['P0'], these_params['rel_permitivity'],
    #                         V0=0, VL=0)

    # def E_field(self):
    #     """Calculate electric field from N, P"""
    #     V = self.voltage()

    #     return E_field_from_V(V, self.inter_dx)

    def delta_n_d(self):
        """Calculate above-equilibrium electron density from N, n0
            inside direct band
        """
        get_these_params = ['N0']
        these_params = self.get_stitched_params(get_these_params)

        return delta_n(self.sim_outputs['N_d'], these_params['N0'])

    def delta_n_ind(self):
        """Calculate above-equilibrium electron density from N, n0
            inside indirect band
        """
        # get_these_params = ['N0']
        # these_params = self.get_stitched_params(get_these_params)

        return delta_n(self.sim_outputs['N_ind'], 0)

    def delta_n(self):
        """Calculate above-equilibrium electron density from N, n0
            total in both bands
        """

        return self.delta_n_d() + self.delta_n_ind()

    def average_delta_n(self):
        node_list = generate_shared_x_array(
            False, self.grid_x_nodes, self.total_lengths)

        temp_dN = self.delta_n()
        temp_dN = intg.trapz(temp_dN, x=node_list, axis=1)
        temp_dN /= sum(self.total_lengths)

        return temp_dN

    def delta_p(self):
        """Calculate above-equilibrium hole density from P, p0"""
        get_these_params = ['P0']
        these_params = self.get_stitched_params(get_these_params)
        return delta_p(self.sim_outputs['P'], these_params['P0'])

    def radiative_recombination_d(self):
        """Calculate radiative recombination (from direct band)"""
        get_these_params = ['B', 'N0', 'P0']
        these_params = self.get_stitched_params(get_these_params)

        return radiative_recombination(self.sim_outputs['N_d'],
                                       self.sim_outputs['P'],
                                       these_params['B'],
                                       these_params['N0'],
                                       these_params['P0'])

    def radiative_recombination_ind(self):
        """Calculate radiative recombination (from direct band)"""
        get_these_params = ['B_ind', 'N0', 'P0']
        these_params = self.get_stitched_params(get_these_params)

        return radiative_recombination(self.sim_outputs['N_ind'],
                                       self.sim_outputs['P'],
                                       these_params['B_ind'],
                                       these_params['N0'],
                                       these_params['P0'])

    def radiative_recombination(self):
        """Calculate radiative recombination (from direct band)"""

        return self.radiative_recombination_d() + \
            self.radiative_recombination_ind()

    def nonradiative_recombination(self):
        """Calculate nonradiative recombination using SRH model
        Assumes quasi steady state trap level occupation."""
        get_these_params = ['N0', 'P0', 'tau_N', 'tau_P']
        these_params = self.get_stitched_params(get_these_params)

        return nonradiative_recombination(self.sim_outputs['N_d'],
                                          self.sim_outputs['P'],
                                          these_params['N0'],
                                          these_params['P0'],
                                          these_params['tau_N'],
                                          these_params['tau_P']
                                          )

    def trap(self):
        get_these_params = ['N0', 'tau_C']
        these_params = self.get_stitched_params(get_these_params)
        return trap(self.sim_outputs['N_d'], these_params['N0'],
                    these_params['tau_C'])

    def detrap(self):
        get_these_params = ['tau_E']
        these_params = self.get_stitched_params(get_these_params)
        return detrap(self.sim_outputs['N_ind'], these_params['tau_E'])

    def PL_d(self):
        """ For more complex systems we would need to do something to the RR
            e.g. waveguiding, FRET
            but wrapping the raw RR will suffice here
        """
        PL_base = self.radiative_recombination_d()
        return PL_base

    def PL_ind(self):
        PL_base = self.radiative_recombination_ind()
        return PL_base

    def PL(self):
        PL_base = self.radiative_recombination()
        return PL_base

    def PL_integral(self, PL_base):
        integral_PL = np.zeros_like(PL_base[:, 0])
        left = right = 0
        for i in range(len(self.layer_names)):
            right += len(self.grid_x_nodes[i])
            thickness = self.total_lengths[i]
            dx_len = self.grid_x_edges[i][1] - self.grid_x_edges[i][0]
            integral_PL += new_integrate(PL_base[:, left:right], 0, thickness,
                                         dx_len, thickness, False)

        return integral_PL

import numpy as np
from scipy import integrate as intg
from utils import to_array
import tables

from Modules.module_Si_dualband.simulations_dydt import dydt_indirect


class SimulationSettings(object):
    def __init__(self, nt, dt, hmax_, rtol_, atol_):
        """
        Various settings for the ODEINT solver

        Parameters
        ----------
        nt : int
            Number of time steps.
        dt : float
            time step size.
        hmax_ : float
            Max internal time step used by solver.
        rtol_ : float
            Relative solver tolerance.
        atol_ : float
            Absolute solver tolerance.

        Returns
        -------
        None.

        """

        self.hmax_ = hmax_
        self.nt = nt
        self.dt = dt
        self.tSteps = np.linspace(0, nt*dt, nt+1)
        self.rtol_ = rtol_
        self.atol_ = atol_


class SpaceGrid():
    def __init__(self):
        pass


class SystemParams():
    def __init__(self):
        pass


class CalculatedValues():
    def __init__(self):
        pass


class OdeSiSimulation():

    def __init__(self, layers: dict, m: dict, flags: dict,
                 init_conditions: dict):

        self.absorber = layers["Absorber"]

        self.m = m["Absorber"]
        self.set_flags(flags)
        self.set_init_conditions(init_conditions)

    def set_flags(self, flags: dict):
        self.do_ss = flags.get('check_do_ss', False)
        self.write_output = flags.get('write_output', True)

    def set_init_conditions(self, init_conditions: dict):
        self.init_N_d = init_conditions.get("N_d", 0)
        self.init_N_ind = init_conditions.get("N_ind", 0)
        self.init_P = init_conditions.get("P", 0)

    def simulate(self, data_path: str, nt: int, dt: float, hmax_=0,
                 rtol_=1e-5, atol_=1e-8):
        """
        Master function for PN_Junction module simulation.
        Problem statement:
        Create a discretized, time and space dependent solution (N(z,t), P(z,t))
        of a pn-junction carrier model with mapi_node_number, rubrene_node_number space steps and time_steps time steps
        Space step size is mapi_node_width, rubrene_node_width; time step is dt
        Initial conditions: self.init_N, self.init_P

        Returns
        -------
        None
            TEDs does not do anything with the return value. Other applications might find this useful however.
        """

        sim = SimulationSettings(nt, dt, hmax_,
                                 rtol_, atol_)

        grid = SpaceGrid()
        grid = self.package_space_grid(grid)

        # Unpack params; typecast non-array params to arrays if needed
        par = SystemParams()
        par = self.unpack_params(grid, par)

        # Package initial condition
        calc = CalculatedValues()
        calc = self.calculate_system_values(grid, par, calc)

        solution = self.simulate_basic(sim, grid, par, calc)

        data = solution.y.T

        if self.write_output:
            self.write_output_to_file(data_path, data, calc)

        return

    def package_space_grid(self, g):
        g.thicknesses = [self.absorber.total_length]  # [nm]
        g.nx = [self.m]

        g.num_layers = len(g.thicknesses)
        g.n_total = np.sum(g.nx)

        g.bounds = np.hstack(([0], np.cumsum(g.thicknesses)))
        g.nx_bounds = np.hstack(([0], np.cumsum(g.nx)))
        g.xFaces = np.hstack(
            [0] + [np.linspace(g.bounds[i], g.bounds[i+1], g.nx[i]+1)[1:] for i in range(g.num_layers)])

        # Using a finite volume approach - nodes can be of variable size
        g.dx = np.diff(g.xFaces)
        g.inter_dx = ((g.dx + np.roll(g.dx, -1))/2)[:-1]

        g.xSteps = g.xFaces[:-1] + g.dx / 2
        return g

    # def stitch_arrays(self, arrays, edge=False):
    #     # This extends the behavior of to_array to suit the PN-junction's
    #     # shared parameters, which are best represented as one long array instead
    #     # of three separate arrays for the layers
    #     # This should be moved to utils.py if it becomes popular in other modules
    #     if edge:
    #         # Set overlapping edges to averages of both layers
    #         for i in range(1, len(arrays)):
    #             arrays[i][0] = (arrays[i][0] + arrays[i-1][-1]) / 2
    #             arrays[i-1] = arrays[i-1][:-1]
    #     return np.hstack(arrays)

    def unpack_params(self, g, p):

        p.RS = [self.absorber.params["Sf"].value,
                self.absorber.params["Sb"].value]

        # Translate from user-friendly displayed parameter names
        # to attribute names used by solver
        solver_param_names = {"mu_N": "mu_n_d", "mu_N_ind": "mu_n_ind",
                              "mu_P": "mu_p", "N0": "n0", "P0": "p0",
                              "tau_C": "tauC", "tau_E": "tauE",
                              "B": "B", "B_ind": "B_ind",
                              "tau_N": "tauN", "tau_P": "tauP",
                              "CN": "Cn", "CP": "Cp",
                              "Temp": "T",
                              }
        for param_name, translated_name in solver_param_names.items():
            setattr(p, translated_name, to_array(
                self.absorber.params[param_name].value, g.nx[0],
                self.absorber.params[param_name].is_edge))

        # setattr(p, p_attrs[param_name],
        #         self.stitch_arrays([to_array(layer.params[param_name].value, g.nx[i], layer.params[param_name].is_edge)
        #                             for i, layer in enumerate(self.layers)], self.layers[0].params[param_name].is_edge)
        #         )

        # p.dEgdx = (np.roll(p.Eg, -1) - p.Eg)[:-1] / g.inter_dx
        # p.dchidx = (np.roll(p.chi, -1) - p.chi)[:-1] / g.inter_dx

        p.V0 = 0
        p.VL = 0
        p.t_old = 0
        return p

    def calculate_system_values(self, g, p, s):
        s = self.make_steady_state_injection(p, s)

        s.init_condition = [self.init_N_d, self.init_N_ind, self.init_P]
        s.data_splits = np.cumsum([len(d) for d in s.init_condition])[:-1]
        s.init_condition = np.hstack(s.init_condition)

        return s

    def make_steady_state_injection(self, p, s):
        if self.do_ss:
            s.inject_dN = self.init_N - p.n0
            s.inject_dP = self.init_P - p.p0

        else:
            s.inject_dN = 0
            s.inject_dP = 0
        return s

    def simulate_basic(self, sim, g, p, s):
        args = (g, p, s, self.do_ss)
        return intg.solve_ivp(dydt_indirect,
                              [0, sim.nt * sim.dt],
                              s.init_condition, args=args, t_eval=sim.tSteps,
                              method='LSODA', max_step=sim.hmax_,
                              rtol=sim.rtol_, atol=sim.atol_)

    def write_output_to_file(self, data_path: str, data: any, s):
        N_d, N_ind, P = np.split(data, s.data_splits, axis=1)
        to_write = {"N_d": N_d, "N_ind": N_ind, "P": P}

        atom = tables.Float64Atom()

        for oname, output in to_write.items():
            with tables.open_file(data_path + f"-{oname}.h5", mode='a') as ofstream:
                table = ofstream.create_earray(
                    ofstream.root, "data", atom, (0, len(output[0])))
                table.append(output)

        return

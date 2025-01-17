import numpy as np
from scipy import integrate as intg
from utils import to_array
import tables

from Modules.module_MAPI_Rubrene_DBP.simulations_dydt import dydt_sct
from Modules.module_MAPI_Rubrene_DBP.simulations_dydt import dydt_basic
from Modules.module_MAPI_Rubrene_DBP.calculations import E_field
from Modules.module_MAPI_Rubrene_DBP.calculations import SST


class SimulationSettings(object):
    def __init__(self, time_step_number , time_step_size, hmax_):
        # hmax_: float = 0              # Maximum internal step size to be taken by ODEINT
        # time_step_number: int             # Number of time steps
        # time_step_size: float       # Time step size
        # data_path: str         # Output file location.
        self.hmax_ = hmax_
        self.time_step_number = time_step_number
        self.time_step_size = time_step_size
        # Do time_step_number time steps
        self.tSteps = np.linspace(0,
                                time_step_number*time_step_size,
                                time_step_number+1)

class SpaceGrid():
    def __init__(self):
        pass

class SystemParams():
    def __init__(self):
        pass

class CalculatedValues():
    def __init__(self):
        pass

class OdeTwoLayerSimulation():
    mapi_node_number: int       # Number of MAPI layer space nodes
    mapi_node_width: float      # MAPI Space node width
    rubrene_node_number: int    # Number of Rubrene layer space nodes
    rubrene_node_width: float   # Rubrene Space node width
    mapi_params: dict           # {"str":Parameter} Collection of parameter objects for MAPI layer
    rubrene_params: dict        # dict {"str":Parameter} Collection of parameter objects for Rubrene layer
    do_fret: bool        # optional, Whether to include the FRET integral
    do_ss: bool           # Whether to inject the initial conditions at every time step, creating a nonzero steady state situation
    no_upconverter: bool  # Whether to block new triplets from being formed at the MAPI/Rubrene interface, which effectively deactivates the latter upconverter layer.
    predict_sstriplets: bool
    # Whether to start the triplet density at a predicted steady state value rather than zero.
    # This reduces the simulation time needed to reach steady state.
    # This only works if self.do_ss is also active
    do_seq_charge_transfer: bool  #
    write_output: bool     # Whether to write output files. TEDs always does this but other applications reusing this function might not
    init_N = 0                    # 1D ndarray, Initial excited electron distribution
    init_P = 0                    # 1D ndarray, Initial hole distribution
    init_T = 0
    init_S = 0
    init_D = 0
    init_P_up = 0

    def __init__(self, mapi_layer: any, rubrene_layer: any,
                m: dict, flags: dict, init_conditions: dict):
        self.mapi_node_width = mapi_layer.dx
        self.mapi_params = mapi_layer.params
        self.rubrene_node_width = rubrene_layer.dx
        self.rubrene_params = rubrene_layer.params

        self.set_node_number(m)
        self.set_flags(flags)
        self.set_init_conditions(init_conditions)
        
    def set_node_number(self, m: dict):
        self.mapi_node_number = m["MAPI"]
        self.rubrene_node_number = m["Rubrene"]

    def set_flags(self, flags: dict):
        self.do_fret = flags.get('do_fret', False)
        self.do_ss = flags.get('check_do_ss', False)
        self.no_upconverter = flags.get('no_upconverter', False)
        self.predict_sstriplets = flags.get('predict_sst', False)
        self.do_seq_charge_transfer = flags.get('do_sct', False)
        self.write_output = flags.get('write_output', True)

    def set_init_conditions(self, init_conditions: dict):
        self.init_N = init_conditions.get("N", 0)
        self.init_P = init_conditions.get("P", 0)
        self.init_T = init_conditions.get("T", 0)
        self.init_S = init_conditions.get("delta_S", 0)
        self.init_D = init_conditions.get("delta_D", 0)
        if self.do_seq_charge_transfer: self.init_P_up = init_conditions.get("P_up", 0)


    def simulate(self, data_path: str, time_step_number: int, time_step_size: float, hmax_ = 0,
                 rtol_=1e-5, atol_=1e-8):
        """
        Master function for MAPI_Rubrene_DBP module simulation.
        Problem statement:
        Create a discretized, time and space dependent solution (N(x,t), P(x,t), T(x,t), S(x,t), D(x,t))
        of a MAPI-Rubrene:DBP carrier model with mapi_node_number, rubrene_node_number space steps and time_steps time steps
        Space step size is mapi_node_width, rubrene_node_width; time step is time_step_size
        Initial conditions: self.init_N, self.init_P, init_T, init_S, init_D
        Optional FRET integral term

        Returns
        -------
        None
            TEDs does not do anything with the return value. Other applications might find this useful however.
        """

        sim = SimulationSettings(time_step_number, time_step_size, hmax_)
        
        grid = SpaceGrid()
        grid = self.package_space_grid(grid)
    
        ## Unpack params; typecast non-array params to arrays if needed
        par = SystemParams()
        par = self.unpack_params(grid, par)

        ## Package initial condition
        calc = CalculatedValues()
        calc = self.calculate_system_values(grid, par, calc)

        if self.do_seq_charge_transfer:
            solution = self.simulate_sct(sim, grid, par, calc)
        else:
            solution = self.simulate_basic(sim, grid, par, calc)
            
        data = solution.y.T

        if self.write_output:
            self.write_output_to_file(data_path, data, calc)

        return

    def package_space_grid(self, g):
        g.mapi_dx = self.mapi_node_width
        g.mapi_nx = self.mapi_node_number
        g.rubrene_dx = self.rubrene_node_width
        g.rubrene_nx = self.rubrene_node_number
        return g
    
    def unpack_params(self,g, p):
        #MAPI
        p.Sf = self.mapi_params["Sf"].value
        p.Sb = self.mapi_params["Sb"].value
        p.mu_n = to_array(self.mapi_params["mu_N"].value, g.mapi_nx, True)
        p.mu_p = to_array(self.mapi_params["mu_P"].value, g.mapi_nx, True)
        p.mapi_temperature = to_array(self.mapi_params["MAPI_temperature"].value, g.mapi_nx, True)
        p.n0 = to_array(self.mapi_params["N0"].value, g.mapi_nx, False)
        p.p0 = to_array(self.mapi_params["P0"].value, g.mapi_nx, False)
        p.B = to_array(self.mapi_params["B"].value, g.mapi_nx, False)
        p.Cn = to_array(self.mapi_params["Cn"].value, g.mapi_nx, False)
        p.Cp = to_array(self.mapi_params["Cp"].value, g.mapi_nx, False)
        p.tauN = to_array(self.mapi_params["tau_N"].value, g.mapi_nx, False)
        p.tauP = to_array(self.mapi_params["tau_P"].value, g.mapi_nx, False)
        p.eps = to_array(self.mapi_params["rel_permitivity"].value, g.mapi_nx, True)
    
        #Rubrene
        p.mu_s = to_array(self.rubrene_params["mu_S"].value, g.rubrene_nx, True)
        p.mu_T = to_array(self.rubrene_params["mu_T"].value, g.rubrene_nx, True)
        p.rubrene_temperature = to_array(self.rubrene_params["Rubrene_temperature"].value, g.rubrene_nx, True)
        p.T0 = to_array(self.rubrene_params["T0"].value, g.rubrene_nx, False)
        p.tauT = to_array(self.rubrene_params["tau_T"].value, g.rubrene_nx, False)
        p.tauS = to_array(self.rubrene_params["tau_S"].value, g.rubrene_nx, False)
        p.tauD = to_array(self.rubrene_params["tau_D"].value, g.rubrene_nx, False)
        p.k_fusion = to_array(self.rubrene_params["k_fusion"].value, g.rubrene_nx, False)
        p.k_0 = to_array(self.rubrene_params["k_0"].value, g.rubrene_nx, False)
        

        if self.do_seq_charge_transfer:
            # Unpack additional params for this physics model
            p.Sp = 0 if self.no_upconverter else self.rubrene_params["Sp"].value
            p.Ssct = 0 if self.no_upconverter else self.rubrene_params["Ssct"].value
            p.w_vb = self.rubrene_params["W_VB"].value
            p.mu_p_up = to_array(self.rubrene_params["mu_P_up"].value, g.rubrene_nx, True)
            p.uc_eps = to_array(self.rubrene_params["uc_permitivity"].value, g.rubrene_nx, True)
        else:
            p.St = 0 if self.no_upconverter else self.rubrene_params["St"].value
        
        return p

    def calculate_system_values(self, g, p, s):
        s = self.make_fret_weights(g, s)
        
        s = self.make_steady_state_injection(p, s)
            
        init_T, init_S, init_D = self.apply_SST_approx(g, p, s)
        
        # An unfortunate workaround - create temporary dictionaries out of necessary values to match the call signature of E_field()
        init_E_field = E_field({"N":self.init_N, "P":self.init_P}, 
                            {"rel_permitivity":p.eps, "N0":p.n0, "P0":p.p0, "Node_width":g.mapi_dx})
        
        init_E_upc = np.zeros(g.rubrene_nx+1)
        
        if self.do_seq_charge_transfer:
            s.init_condition = [self.init_N, self.init_P, init_E_field, init_T, init_S, init_D, self.init_P_up, init_E_upc]
            s.data_splits = np.cumsum([len(d) for d in s.init_condition])[:-1]            
            s.init_condition = np.concatenate(s.init_condition, axis=None)
        else:
            s.init_condition = [self.init_N, self.init_P, init_E_field, init_T, init_S, init_D]
            s.data_splits = np.cumsum([len(d) for d in s.init_condition])[:-1] 
            s.init_condition = np.concatenate(s.init_condition, axis=None)
        
        return s
    
    def make_fret_weights(self, g, s):
        if self.do_fret:
            init_m = np.linspace(g.mapi_dx / 2, g.mapi_nx*g.mapi_dx - g.mapi_dx / 2, g.mapi_nx)
            init_f = np.linspace(g.rubrene_dx / 2, g.rubrene_nx*g.rubrene_dx - g.rubrene_dx / 2, g.rubrene_nx)
            s.wt_fret_to_mapi = np.array([1 / ((init_f + (g.mapi_nx*g.mapi_dx - i)) ** 3) for i in init_m])
            s.wt_fret_from_rubrene = np.array([1 / ((i + (g.mapi_nx*g.mapi_dx - init_m)) ** 3) for i in init_f])
            # It turns out that weight2 contains ALL of the parts of the FRET integral that depend
            # on the variable of integration, so we do that integral right away.
            s.wt_fret_from_rubrene = intg.trapz(s.wt_fret_from_rubrene, dx=g.mapi_dx, axis=1)

        else:
            s.wt_fret_to_mapi = 0
            s.wt_fret_from_rubrene = 0
        return s

    def make_steady_state_injection(self, p, s):
        if self.do_ss:
            s.init_dN = self.init_N - p.n0
            s.init_dP = self.init_P - p.p0
            
        else:
            s.init_dN = 0
            s.init_dP = 0
        return s
    
    def apply_SST_approx(self, g, p, s):
        init_T = self.init_T
        init_S = self.init_S
        init_D = self.init_D
            
        if self.do_ss and self.predict_sstriplets:
            print("Overriding init_T")
            try:
                np.testing.assert_almost_equal(s.init_dN, s.init_dP)
            except AssertionError:
                print("Warning: ss triplet prediction assumes equal excitation of holes and electrons. Unequal excitation is WIP.")
                
            if self.do_fret:
                s.tauD_eff = (p.k_0 * s.wt_fret_from_rubrene / p.tauD) + (1/p.tauD)
            else:
                p.tauD_eff = 1 / p.tauD
                
            if self.do_seq_charge_transfer:
                #raise NotImplementedError
                print("Warning: SST not implemented for seq charge transfer model. Keeping original triplet count")
                
            else:
                init_T, init_S, init_D = SST(p.tauN[-1], p.tauP[-1], p.n0[-1], p.p0[-1], p.B[-1], 
                                            p.St, p.k_fusion, p.tauT, p.tauS, p.tauD_eff, g.rubrene_nx*g.rubrene_dx, np.mean(s.init_dN))
            
        return init_T, init_S, init_D
    
    def simulate_sct(self, sim, g, p, s):
        args=(g, p, s, self.do_fret, self.do_ss)
        return intg.solve_ivp(dydt_sct,
                            [0, sim.time_step_number * sim.time_step_size],
                            s.init_condition, args=args, t_eval=sim.tSteps,
                            method='BDF', max_step=sim.hmax_)   #  Variable time_step_size explicit

    def simulate_basic(self, sim, g, p, s):
        args=(g, p, s, self.do_fret, self.do_ss)
        return intg.solve_ivp(dydt_basic,
                            [0,sim.time_step_number * sim.time_step_size],
                            s.init_condition, args=args, t_eval=sim.tSteps,
                            method='BDF', max_step=sim.hmax_)   #  Variable time_step_size explicit 

    def write_output_to_file(self, data_path: str, data: any, s):
        if self.do_seq_charge_transfer:
            N, P, E_field, T, S, D, P_up, E_up = np.split(data, s.data_splits, axis=1)
            to_write = {"N":N, "P":P, "T":T, "delta_S":S, "delta_D":D, "P_up":P_up}
            
        else:
            N, P, E_field, T, S, D = np.split(data, s.data_splits, axis=1)
            to_write = {"N":N, "P":P, "T":T, "delta_S":S, "delta_D":D}

        atom = tables.Float64Atom()
        
        for oname, output in to_write.items():
            with tables.open_file(data_path + f"-{oname}.h5", mode='a') as ofstream:
                table = ofstream.create_earray(ofstream.root, "data", atom, (0, len(output[0])))
                table.append(output)
            
        return
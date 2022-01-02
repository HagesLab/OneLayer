# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
import numpy as np
import os
from scipy import integrate as intg
from scipy import optimize
from helper_structs import Parameter, Output, Layer
from utils import u_read, to_index, to_array, to_pos, new_integrate
import tables
from _OneD_Model import OneD_Model
from Modules.module_MAPI_Rubrene_DBP.dydt_sct import dydt_sct
from Modules.module_MAPI_Rubrene_DBP.definitions import define_layers
from Modules.module_MAPI_Rubrene_DBP.definitions import define_flags
from Modules.module_MAPI_Rubrene_DBP.initializations import MAPI_Rubrene_Initial_Conditions


q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]



class MAPI_Rubrene(OneD_Model):
    # A Nanowire object stores all information regarding the initial state being edited in the IC tab
    # And functions for managing other previously simulated nanowire data as they are loaded in
    def __init__(self):
        super().__init__()
        self.system_ID = "MAPI_Rubrene"
        self.time_unit = "[ns]"
        self.flags_dict = define_flags()
        self.layers = define_layers()

        return


    def calc_inits(self):
        """Calculate initial electron and hole density distribution"""
        
        mapi = self.layers["MAPI"]
        rubrene = self.layers["Rubrene"]

        mapi_rubrene_inits = MAPI_Rubrene_Initial_Conditions(mapi, rubrene)

        return mapi_rubrene_inits.format_inits_to_dict()


    def simulate(self, data_path, m, n, dt, flags, hmax_, init_conditions):
        """Calls ODEINT solver."""
        mapi = self.layers["MAPI"]
        ru = self.layers["Rubrene"]
        for param_name, param in mapi.params.items():
            param.value *= mapi.convert_in[param_name]
            
        for param_name, param in ru.params.items():
            param.value *= ru.convert_in[param_name]

        ode_twolayer(data_path, m["MAPI"], mapi.dx, m["Rubrene"], ru.dx,
                     n, dt, mapi.params, ru.params, flags['do_fret'].value(),
                     flags['check_do_ss'].value(), flags['no_upconverter'].value(),
                     flags['predict_sst'].value(), flags["do_sct"].value(),
                     hmax_, True,
                     init_conditions["N"], init_conditions["P"],
                     init_conditions["T"], init_conditions["delta_S"],
                     init_conditions["delta_D"], init_conditions["P_up"])
    
    def get_overview_analysis(self, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
        """Calculates at a selection of sample times: N, P, (total carrier densities)
           delta_N, delta_P, (above-equilibrium carrier densities)
           internal electric field due to differences in N, P,
           radiative recombination,
           non-radiative (SRH model) recombination,
           
           Integrates over nanowire length: PL due to radiative recombination, waveguiding, and carrier regeneration"""
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        mapi = self.layers["MAPI"]
        mapi_params = params["MAPI"]
        mapi_length = mapi_params["Total_length"]
        dm = mapi_params["Node_width"]
        
        ru = self.layers["Rubrene"]
        ru_params = params["Rubrene"]
        ru_length = ru_params["Total_length"]
        df = ru_params["Node_width"]
        
        data_dict = {"MAPI":{}, "Rubrene":{}}
        
        for layer_name, layer in self.layers.items():
            for raw_output_name in layer.s_outputs:
                data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, 
                                                     raw_output_name)
                data = []
                for tstep in tsteps:
                    data.append(u_read(data_filename, t0=tstep, single_tstep=True))
                
                data_dict[layer_name][raw_output_name] = np.array(data)
                    
        data_dict["MAPI"]["E_field"] = E_field(data_dict["MAPI"], mapi_params)
        data_dict["MAPI"]["delta_N"] = delta_n(data_dict["MAPI"], mapi_params)
        data_dict["MAPI"]["delta_P"] = delta_p(data_dict["MAPI"], mapi_params)
        data_dict["MAPI"]["RR"] = radiative_recombination(data_dict["MAPI"], mapi_params)
        data_dict["MAPI"]["NRR"] = nonradiative_recombination(data_dict["MAPI"], mapi_params)
                
        #### MAPI PL ####
        with tables.open_file(os.path.join(data_dirname, file_name_base + "-N.h5"), mode='r') as ifstream_N, \
            tables.open_file(os.path.join(data_dirname, file_name_base + "-P.h5"), mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data)
            temp_P = np.array(ifstream_P.root.data)
        temp_RR = radiative_recombination({"N":temp_N, "P":temp_P}, mapi_params)
        PL_base = prep_PL(temp_RR, 0, to_index(mapi_length, dm, mapi_length), 
                          False, mapi_params, "MAPI")
        data_dict["MAPI"]["mapi_PL"] = new_integrate(PL_base, 0, mapi_length, 
                                                    dm, mapi_length, False)
        
        temp_dN = delta_n({"N":temp_N}, mapi_params)
        temp_dN = intg.trapz(temp_dN, dx=dm, axis=1)
        temp_dN /= mapi_length
        data_dict["MAPI"]["avg_delta_N"] = temp_dN
        
        try:
            data_dict["MAPI"]["tau_diff"] = tau_diff(data_dict["MAPI"]["mapi_PL"], dt)
        except Exception:
            print("Error: failed to calculate tau_diff")
            data_dict["MAPI"]["tau_diff"] = 0
        #################
        
        data_dict["Rubrene"]["E_upc"] = E_field_r(data_dict["Rubrene"], ru_params)
        
        #### DBP PL ####
        with tables.open_file(os.path.join(data_dirname, file_name_base + "-delta_D.h5"), mode='r') as ifstream_D:
            temp_D = np.array(ifstream_D.root.data)
            
        temp_D = prep_PL(temp_D, 0, to_index(ru_length, df, ru_length), 
                          False, ru_params, "Rubrene")
    
        data_dict["Rubrene"]["dbp_PL"] = new_integrate(temp_D, 0, ru_length, 
                                                       df, ru_length, False)
        ################
        
        #### TTA Rate ####
        # "Triplet-Triplet Annihilation"
        with tables.open_file(os.path.join(data_dirname, file_name_base + "-T.h5"), mode='r') as ifstream_T:
            temp_TTA = np.array(ifstream_T.root.data)
            
        temp_TTA = TTA(temp_TTA, 0, to_index(ru_length, df, ru_length), 
                       False, ru_params)
    
        data_dict["Rubrene"]["TTA"] = new_integrate(temp_TTA, 0, ru_length, 
                                                    df, ru_length, False)
        ##################
        
        #### Efficiencies ####
        with tables.open_file(os.path.join(data_dirname, file_name_base + "-N.h5"), mode='r') as ifstream_N, \
             tables.open_file(os.path.join(data_dirname, file_name_base + "-P.h5"), mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data[:,-1])
            temp_init_N = np.array(ifstream_N.root.data[0,:])
            temp_P = np.array(ifstream_P.root.data[:,-1])
                
            temp_init_N = intg.trapz(temp_init_N, dx=dm)
            
            
        tail_n0 = mapi_params["N0"]
        if isinstance(tail_n0, np.ndarray):
            tail_n0 = tail_n0[-1]
            
        tail_p0 = mapi_params["P0"]
        if isinstance(tail_p0, np.ndarray):
            tail_p0 = tail_p0[-1]
            
        if "no_upconverter" in flags and flags["no_upconverter"]:
            t_form = temp_N * 0
            
        else:
            t_form = ru_params["St"] * ((temp_N * temp_P - tail_n0 * tail_p0)
                                      / (temp_N + temp_P))
        
        try:
            data_dict["Rubrene"]["T_form_eff"] =  t_form / temp_init_N
        except FloatingPointError:
            data_dict["Rubrene"]["T_form_eff"] =  t_form * 0
        try:
            data_dict["Rubrene"]["T_anni_eff"] = data_dict["Rubrene"]["TTA"] / t_form
        except FloatingPointError:
            data_dict["Rubrene"]["T_anni_eff"] = data_dict["Rubrene"]["TTA"] * 0
            
            
        data_dict["Rubrene"]["S_up_eff"] = data_dict["Rubrene"]["T_anni_eff"] * data_dict["Rubrene"]["T_form_eff"]
        
        try:
            data_dict["MAPI"]["eta_MAPI"] = data_dict["MAPI"]["mapi_PL"] / temp_init_N
        except FloatingPointError:
            data_dict["MAPI"]["eta_MAPI"] = data_dict["MAPI"]["mapi_PL"] * 0
            
        try:
            data_dict["Rubrene"]["eta_UC"] = data_dict["Rubrene"]["dbp_PL"] / temp_init_N
        except FloatingPointError:
            data_dict["Rubrene"]["eta_UC"] = data_dict["Rubrene"]["dbp_PL"] * 0
            
        for data in data_dict["MAPI"]:
            data_dict["MAPI"][data] *= mapi.convert_out[data]
            
        for data in data_dict["Rubrene"]:
            data_dict["Rubrene"][data] *= ru.convert_out[data]
            
        data_dict["MAPI"]["mapi_PL"] *= mapi.iconvert_out["mapi_PL"]
        data_dict["Rubrene"]["dbp_PL"] *= ru.iconvert_out["dbp_PL"]
        data_dict["Rubrene"]["TTA"] *= ru.iconvert_out["TTA"]
        
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, flags, for_integrate=False, 
                     i=0, j=0, nen=False, extra_data = None):
        """ Provides delta_N, delta_P, electric field, recombination, 
            and spatial PL values on demand.
        """
        # For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        where_layer = self.find_layer(datatype)
        
        layer = self.layers[where_layer]
        layer_params = params[where_layer]
        layer_sim_data = sim_data[where_layer]
        data = None
        if (datatype in layer.s_outputs):
            data = layer_sim_data[datatype]
        
        else:
            if (datatype == "delta_N"):
                data = delta_n(layer_sim_data, layer_params)
                
            elif (datatype == "delta_P"):
                data = delta_p(layer_sim_data, layer_params)
                
            elif (datatype == "delta_T"):
                data = delta_T(layer_sim_data, layer_params)
                
            elif (datatype == "RR"):
                data = radiative_recombination(layer_sim_data, layer_params)

            elif (datatype == "NRR"):
                data = nonradiative_recombination(layer_sim_data, layer_params)
                
            elif (datatype == "E_field"):
                data = E_field(layer_sim_data, layer_params)

            elif (datatype == "mapi_PL"):
    
                if for_integrate:
                    rad_rec = radiative_recombination(extra_data[where_layer], layer_params)
                    data = prep_PL(rad_rec, i, j, nen, layer_params, where_layer)
                else:
                    rad_rec = radiative_recombination(layer_sim_data, layer_params)
                    data = prep_PL(rad_rec, 0, len(rad_rec)-1, False, 
                                   layer_params, where_layer).flatten()
                    
            elif (datatype == "E_upc"):
                data = E_field_r(layer_sim_data, layer_params)
                    
            elif (datatype == "dbp_PL"):
    
                if for_integrate:
                    delta_D = extra_data[where_layer]["delta_D"]
                    data = prep_PL(delta_D, i, j, nen, layer_params, where_layer)
                else:
                    delta_D = layer_sim_data["delta_D"]
                    data = prep_PL(delta_D, 0, len(delta_D)-1, False, 
                                   layer_params, where_layer).flatten()
                    
            elif (datatype == "TTA"):
    
                if for_integrate:
                    T = extra_data[where_layer]["T"]
                    data = TTA(T, i, j, nen, layer_params)
                else:
                    T = layer_sim_data["T"]
                    data = TTA(T, 0, len(T)-1, False, 
                               layer_params).flatten()
                    
            
            else:
                raise ValueError
                
        return data
    
    def get_timeseries(self, pathname, datatype, parent_data, total_time, dt, params, flags):
        
        if datatype == "delta_N":
            temp_dN = parent_data / params["MAPI"]["Total_length"]
            return [("avg_delta_N", temp_dN)]
        
        if datatype == "mapi_PL":
            # photons emitted per photon absorbed
            # generally this is meaningful in ss mode - thus units are [phot/nm^2 ns] per [carr/nm^2 ns]
            with tables.open_file(pathname + "-N.h5", mode='r') as ifstream_N:
                temp_init_N = np.array(ifstream_N.root.data[0,:])
                
            temp_init_N = intg.trapz(temp_init_N, dx=params["MAPI"]["Node_width"])
            try:
                tdiff = tau_diff(parent_data, dt)
            except FloatingPointError:
                print("Error: failed to calculate tau_diff - effective lifetime is near infinite")
                tdiff = np.zeros_like(np.linspace(0, total_time, int(total_time/dt) + 1))
                
            return [("tau_diff", tdiff),
                    ("eta_MAPI", parent_data / temp_init_N)]
                    
        
        elif datatype == "TTA":
            with tables.open_file(pathname + "-N.h5", mode='r') as ifstream_N, \
                tables.open_file(pathname + "-P.h5", mode='r') as ifstream_P:
                temp_N = np.array(ifstream_N.root.data[:,-1])
                temp_init_N = np.array(ifstream_N.root.data[0,:])
                temp_P = np.array(ifstream_P.root.data[:,-1])
                
            temp_init_N = intg.trapz(temp_init_N, dx=params["MAPI"]["Node_width"])

            tail_n0 = params["MAPI"]["N0"]
            if isinstance(tail_n0, np.ndarray):
                tail_n0 = tail_n0[-1]
                
            tail_p0 = params["MAPI"]["P0"]
            if isinstance(tail_p0, np.ndarray):
                tail_p0 = tail_p0[-1]
                
            if "no_upconverter" in flags and flags["no_upconverter"]:
                t_form = 0 * temp_N
            else:
                t_form = params["Rubrene"]["St"] * ((temp_N * temp_P - tail_n0 * tail_p0)
                                                / (temp_N + temp_P))
            
            # In order:
            # Triplets formed per photon absorbed
            try:
                t_form_eff = t_form / temp_init_N
            except FloatingPointError:
                t_form_eff = t_form * 0
            # Singlets formed per triplet formed
            try:
                t_anni_eff = parent_data / t_form
            except FloatingPointError:
                t_anni_eff = parent_data * 0
            # Singlets formed per photon absorbed
            s_up_eff = t_form_eff * t_anni_eff
            return [("T_form_eff", t_form_eff),
                    ("T_anni_eff", t_anni_eff),
                    ("S_up_eff", s_up_eff)]
        
        elif datatype == "dbp_PL":
            with tables.open_file(pathname + "-N.h5", mode='r') as ifstream_N:
                temp_init_N = np.array(ifstream_N.root.data[0,:])
                
            temp_init_N = intg.trapz(temp_init_N, dx=params["MAPI"]["Node_width"])
            
            # DBP photons emitted per photon absorbed
            return [("eta_UC", parent_data / temp_init_N)]
        else:
            return
        
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        """ Set delta_N and delta_P of outgoing regenerated IC file."""
        param_dict = param_dict["MAPI"]
        sim_data = sim_data["MAPI"]
        include_flags = include_flags["MAPI"]
        param_dict["delta_N"] = (sim_data["N"] - param_dict["N0"]) if include_flags['N'] else np.zeros(len(grid_x))
                    
        param_dict["delta_P"] = (sim_data["P"] - param_dict["P0"]) if include_flags['P'] else np.zeros(len(grid_x))

        return
    

def dydt(t, y, m, f, dm, df, Cn, Cp, 
         tauN, tauP, tauT, tauS, tauD, 
         mu_n, mu_p, mu_s, mu_T,
         n0, p0, T0, Sf, Sb, St, B, k_fusion, k_0, mapi_temperature, rubrene_temperature,
         eps, weight1=0, weight2=0, do_Fret=False, do_ss=False, 
         init_dN=0, init_dP=0):
    """Derivative function for two-layer carrier model."""
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((m+1))
    Jp = np.zeros((m+1))
    JT = np.zeros((f+1))
    JS = np.zeros((f+1))

    # Unpack simulated variables
    N = y[0:m]
    P = y[m:2*(m)]
    E_field = y[2*(m):3*(m)+1]
    delta_T = y[3*(m)+1:3*(m)+1+f]
    delta_S = y[3*(m)+1+f:3*(m)+1+2*(f)]
    delta_D = y[3*(m)+1+2*(f):]
    
    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    
    # MAPI boundaries
    Sft = Sf * (N[0] * P[0] - n0[0] * p0[0]) / (N[0] + P[0])
    Sbt = Sb * (N[m-1] * P[m-1] - n0[m-1] * p0[m-1]) / (N[m-1] + P[m-1])
    Stt = St * (N[m-1] * P[m-1] - n0[m-1] * p0[m-1]) / (N[m-1] + P[m-1])
    Jn[0] = Sft
    Jn[m] = -(Sbt+Stt)
    Jp[0] = -Sft
    Jp[m] = (Sbt+Stt)
    
    # Rubrene boundaries
    JT[0] = -Stt
    JT[f] = 0
    JS[0] = 0
    JS[f] = 0

    ## Calculate Jn, Jp [nm^-2 ns^-1] for MAPI, 
    Jn[1:-1] = (-mu_n[1:-1] * (N_edges) * (q * E_field[1:-1]) 
                + (mu_n[1:-1]*kB*mapi_temperature[1:-1]) * ((np.roll(N,-1)[:-1] - N[:-1]) / (dm)))

    Jp[1:-1] = (-mu_p[1:-1] * (P_edges) * (q * E_field[1:-1]) 
                - (mu_p[1:-1]*kB*mapi_temperature[1:-1]) * ((np.roll(P, -1)[:-1] - P[:-1]) / (dm)))

    dJn = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dm)
    dJp = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dm)
        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (eps * eps0))
    
    ## Rubrene J fluxes
    JT[1:-1] = (mu_T[1:-1]*kB*rubrene_temperature[1:-1]) * ((np.roll(delta_T,-1)[:-1] - delta_T[:-1]) / (df))
    JS[1:-1] = (mu_s[1:-1]*kB*rubrene_temperature[1:-1]) * ((np.roll(delta_S,-1)[:-1] - delta_S[:-1]) / (df))
    dJT = (np.roll(JT, -1)[:-1] - JT[:-1]) / (df)
    dJS = (np.roll(JS, -1)[:-1] - JS[:-1]) / (df)
    
    ## Calculate recombination (consumption) terms
    # MAPI Auger + RR + SRH
    n_rec = (Cn*N + Cp*P + B + (1 / ((tauN * P) + (tauP * N)))) * (N * P - n0 * p0)
    p_rec = n_rec
        
    # Rubrene single- and bi-molecular decays
    T_rec = delta_T / tauT
    T_fusion = k_fusion * (delta_T) ** 2
    S_Fret = delta_S / tauS
    D_rec = delta_D / tauD
    
    ## Calculate D_Fretting
    if do_Fret:
        D_Fret1 = intg.trapz(weight1 * delta_D * k_0 / tauD, dx=df, axis=1)
        D_Fret2 = (delta_D * k_0 / tauD) * weight2
        
    else:
        D_Fret1 = 0
        D_Fret2 = 0
    
    dNdt = ((1/q) * dJn - n_rec + D_Fret1)
    if do_ss: 
        dNdt += init_dN

    dPdt = ((1/q) * -dJp - p_rec + D_Fret1)
    if do_ss: 
        dPdt += init_dP
        
    dTdt = ((1/q) * dJT - T_fusion - T_rec)
    dSdt = ((1/q) * dJS + T_fusion - S_Fret)
    dDdt = S_Fret - D_Fret2 - D_rec

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt, dTdt, dSdt, dDdt], axis=None)
    return dydt
    
def ode_twolayer(data_path_name, m, dm, f, df, n, dt, mapi_params, ru_params, 
                 do_Fret=False, do_ss=False, no_upconverter=False, predict_sstriplets=False,
                 do_seq_charge_transfer=False,
                 hmax_=0, write_output=True, 
                 init_N=0, init_P=0, init_T=0, init_S=0, init_D=0, init_P_up=0):
    """
    Master function for MAPI_Rubrene_DBP module simulation.
    Problem statement:
    Create a discretized, time and space dependent solution (N(x,t), P(x,t), T(x,t), S(x,t), D(x,t))
    of a MAPI-Rubrene:DBP carrier model with m, f space steps and n time steps
    Space step size is dm, df; time step is dt
    Initial conditions: init_N, init_P, init_T, init_S, init_D
    Optional FRET integral term

    Parameters
    ----------
    data_path_name : str
        Output file location.
    m : int
        Number of MAPI layer space nodes.
    dm : float
        MAPI Space node width.
    f : int
        Number of Rubrene layer space nodes.
    df : float
        Rubrene Space node width.
    n : int
        Number of time steps.
    dt : float
        Time step size.
    mapi_params : dict {"str":Parameter}
        Collection of parameter objects for MAPI layer
    ru_params : dict {"str":Parameter}
        Collection of parameter objects for Rubrene layer
    do_Fret : bool, optional
        Whether to include the FRET integral. The default is True.
    do_ss : bool, optional
        Whether to inject the initial conditions at every time step, creating a nonzero steady state situation. The default is False.
    no_upconverter : bool, optional
        Whether to block new triplets from being formed at the MAPI/Rubrene interface, which effectively deactivates the latter upconverter layer.
        The default is False.
    predict_sstriplets : bool, optional
        Whether to start the triplet density at a predicted steady state value rather than zero.
        This reduces the simulation time needed to reach steady state.
        This only works if do_ss is also active.
        The default is False.
    hmax_ : float, optional
        Maximum internal step size to be taken by ODEINT. The default is 0.
    write_output : bool, optional
        Whether to write output files. TEDs always does this but other applications reusing this function might not. The default is True.
    init_N : 1D ndarray, optional
        Initial excited electron distribution. The default is 0.
    init_P : 1D ndarray, optional
        Initial hole distribution. The default is 0.

    Returns
    -------
    None
        TEDs does not do anything with the return value. Other applications might find this useful however.
    """
    
    ## Set data type of array files
    atom = tables.Float64Atom()

    ## Unpack params; typecast non-array params to arrays if needed
    Sf = mapi_params["Sf"].value
    Sb = mapi_params["Sb"].value
    if do_seq_charge_transfer:
        # Unpack additional params for this physics model
        Sp = 0 if no_upconverter else ru_params["Sp"].value
        Ssct = ru_params["Ssct"].value
        w_vb = ru_params["W_VB"].value
        mu_p_up = to_array(ru_params["mu_P_up"].value, f, True)
        uc_eps = to_array(ru_params["uc_permitivity"].value, f, True)

    else:
        St = 0 if no_upconverter else ru_params["St"].value

    mu_n = to_array(mapi_params["mu_N"].value, m, True)
    mu_p = to_array(mapi_params["mu_P"].value, m, True)
    mu_s = to_array(ru_params["mu_S"].value, f, True)
    mu_T = to_array(ru_params["mu_T"].value, f, True)
    mapi_temperature = to_array(mapi_params["MAPI_temperature"].value, m, True)
    rubrene_temperature = to_array(ru_params["Rubrene_temperature"].value, f, True)
    n0 = to_array(mapi_params["N0"].value, m, False)
    p0 = to_array(mapi_params["P0"].value, m, False)
    T0 = to_array(ru_params["T0"].value, f, False)
    tauN = to_array(mapi_params["tau_N"].value, m, False)
    tauP = to_array(mapi_params["tau_P"].value, m, False)
    tauT = to_array(ru_params["tau_T"].value, f, False)
    tauS = to_array(ru_params["tau_S"].value, f, False)
    tauD = to_array(ru_params["tau_D"].value, f, False)
    B = to_array(mapi_params["B"].value, m, False)
    Cn = to_array(mapi_params["Cn"].value, m, False)
    Cp = to_array(mapi_params["Cp"].value, m, False)
    k_fusion = to_array(ru_params["k_fusion"].value, f, False)
    k_0 = to_array(ru_params["k_0"].value, f, False)
    eps = to_array(mapi_params["rel_permitivity"].value, m, True)
    
   
    ## Package initial condition
    # An unfortunate workaround - create temporary dictionaries out of necessary values to match the call signature of E_field()
    init_E_field = E_field({"N":init_N, "P":init_P}, 
                           {"rel_permitivity":eps, "N0":n0, "P0":p0, "Node_width":dm})
    
    init_E_upc = np.zeros(f+1)
    #init_E_field = np.zeros(m+1)
    
    ## Generate a weight distribution needed for FRET term
    if do_Fret:
        init_m = np.linspace(dm / 2, m*dm - dm / 2, m)
        init_f = np.linspace(df / 2, f*df - df / 2, f)
        weight1 = np.array([1 / ((init_f + (m*dm - i)) ** 3) for i in init_m])
        weight2 = np.array([1 / ((i + (m*dm - init_m)) ** 3) for i in init_f])
        # It turns out that weight2 contains ALL of the parts of the FRET integral that depend
        # on the variable of integration, so we do that integral right away.
        weight2 = intg.trapz(weight2, dx=dm, axis=1)

    else:
        weight1 = 0
        weight2 = 0
    
    if do_ss:
        init_dN = init_N - n0
        init_dP = init_P - p0
        
    else:
        init_dN = 0
        init_dP = 0
        
    if do_ss and predict_sstriplets:
        print("Overriding init_T")
        try:
            np.testing.assert_almost_equal(init_dN, init_dP)
        except AssertionError:
            print("Warning: ss triplet prediction assumes equal excitation of holes and electrons. Unequal excitation is WIP.")
            
        if do_Fret:
            tauD_eff = (k_0*weight2/tauD) + (1/tauD)
        else:
            tauD_eff = 1 / tauD
        init_T, init_S, init_D = SST(tauN[-1], tauP[-1], n0[-1], p0[-1], B[-1], 
                                     St, k_fusion, tauT, tauS, tauD_eff, f*df, np.mean(init_dN))
    
    
    if do_seq_charge_transfer:
        init_condition = np.concatenate([init_N, init_P, init_E_field, init_T, init_S, init_D, init_P_up, init_E_upc], axis=None)
    else:
        init_condition = np.concatenate([init_N, init_P, init_E_field, init_T, init_S, init_D], axis=None)

    

    ## Do n time steps
    tSteps = np.linspace(0, n*dt, n+1)
    
    
    if do_seq_charge_transfer:
        args=(m, f, dm, df, Cn, Cp, 
                tauN, tauP, tauT, tauS, tauD, 
                mu_n, mu_p, mu_s, mu_T,
                n0, p0, T0, Sf, Sb, B, k_fusion, k_0, mapi_temperature, rubrene_temperature,
                eps, uc_eps,
                mu_p_up, Ssct, Sp, w_vb, 
                weight1, weight2, do_Fret, do_ss, 
                init_dN, init_dP)
        sol = intg.solve_ivp(dydt_sct, [0,n*dt], init_condition, args=args, t_eval=tSteps, method='BDF', max_step=hmax_)   #  Variable dt explicit
    else:
        args=(m, f, dm, df, Cn, Cp, 
                tauN, tauP, tauT, tauS, tauD, 
                mu_n, mu_p, mu_s, mu_T,
                n0, p0, T0, Sf, Sb, St, B, k_fusion, k_0, mapi_temperature, rubrene_temperature,
                eps, weight1, weight2, do_Fret, do_ss, 
                init_dN, init_dP)
        sol = intg.solve_ivp(dydt, [0,n*dt], init_condition, args=args, t_eval=tSteps, method='BDF', max_step=hmax_)   #  Variable dt explicit
    
    data = sol.y.T
    e1 = data[:, 2*m:3*m+1]
    e2 = data[:,3*m+1+4*f:]
    if write_output:
        ## Prep output files
        # TODO: Py 3.9 removes need for \
        with tables.open_file(data_path_name + "-N.h5", mode='a') as ofstream_N, \
                tables.open_file(data_path_name + "-P.h5", mode='a') as ofstream_P,\
                tables.open_file(data_path_name + "-T.h5", mode='a') as ofstream_T,\
                tables.open_file(data_path_name + "-delta_S.h5", mode='a') as ofstream_S,\
                tables.open_file(data_path_name + "-delta_D.h5", mode='a') as ofstream_D,\
                tables.open_file(data_path_name + "-P_up.h5", mode='a') as ofstream_P_up:
            array_N = ofstream_N.root.data
            array_P = ofstream_P.root.data
            array_T = ofstream_T.root.data
            array_S = ofstream_S.root.data
            array_D = ofstream_D.root.data
            array_P_up = ofstream_P_up.root.data
            
            array_N.append(data[1:,0:m])
            array_P.append(data[1:,m:2*(m)])
            array_T.append(data[1:,3*(m)+1:3*(m)+1+f])
            array_S.append(data[1:,3*(m)+1+f:3*(m)+1+2*(f)])
            # array_D.append(data[1:,3*(m)+1+2*(f):])
            array_D.append(data[1:,3*(m)+1+2*(f):3*(m)+1+3*(f)])
            if do_seq_charge_transfer:
                array_P_up.append(data[1:, 3*(m)+1+3*(f):3*m+1 + 4*f])
            else:
                # No hole transfer - Insert dummy zeros
                array_P_up.append(np.zeros_like(data[1:,3*(m)+1:3*(m)+1+f:3*m+1 + 4*f]))

        return #error_data

    else:
        array_N = data[:,0:m]
        array_P = data[:,m:2*(m)]
        array_T = data[1:,3*(m)+1:3*(m)+1+f]
        array_S = data[1:,3*(m)+1+f:3*(m)+1+2*(f)]
        # array_D = data[1:,3*(m)+1+2*(f):]
        array_D = data[1:,3*(m)+1+2*(f):3*(m)+1+3*(f)]

        return #array_N, array_P, error_data
    
def SST(tauN, tauP, n0, p0, B, St, k_fusion, tauT, tauS, tauD_eff, ru_thickness, gen_rate):
    # FIXME: This doesn't work with sequential charge transfer yet
    n_bal = lambda n, src, tn, tp, n0, p0, B: src - (n**2 - n0*p0) * (B + 1/(n * (tn+tp)))
    
    ss_n = optimize.root(n_bal, gen_rate, args=(gen_rate, tauN, tauP, n0, p0, B))
    
    ss_n = ss_n.x # [nm^-3]
    
    T_gen_per_bin = St * (ss_n**2 - n0*p0) / (ss_n + ss_n) / ru_thickness # [nm^-3 ns^-1]
    
    ss_t = 1.05*(-(1/tauT) + np.sqrt(tauT**-2 + 4*k_fusion*T_gen_per_bin)) / (2*k_fusion)
    
    ss_s = k_fusion * tauS * ss_t**2
    ss_d = ss_s * tauD_eff / tauS
    
    return ss_t, ss_s, ss_d
    
def E_field(sim_outputs, params):
    """Calculate electric field from N, P"""
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q_C = 1.602e-19 # [C per carrier]
    if isinstance(params["rel_permitivity"], np.ndarray):
        averaged_rel_permitivity = (params["rel_permitivity"][:-1] + np.roll(params["rel_permitivity"], -1)[:-1]) / 2
    else:
        averaged_rel_permitivity = params["rel_permitivity"]
    
    dEdx = q_C * (delta_p(sim_outputs, params) - delta_n(sim_outputs, params)) / (eps0 * averaged_rel_permitivity)
    if dEdx.ndim == 1:
        E_field = np.concatenate(([0], np.cumsum(dEdx) * params["Node_width"])) #[V/nm]
        #E_field[-1] = 0
    else:
        E_field = np.concatenate((np.zeros(len(dEdx)).reshape((len(dEdx), 1)), np.cumsum(dEdx, axis=1) * params["Node_width"]), axis=1) #[V/nm]
        #E_field[:,-1] = 0
    return E_field

def E_field_r(sim_outputs, params):
    """Calculate electric field from T+S+D, Q"""
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q_C = 1.602e-19 # [C per carrier]
    if isinstance(params["uc_permitivity"], np.ndarray):
        averaged_rel_permitivity = (params["uc_permitivity"][:-1] + np.roll(params["uc_permitivity"], -1)[:-1]) / 2
    else:
        averaged_rel_permitivity = params["uc_permitivity"]
    
    dEdx = q_C * (sim_outputs["P_up"] - (sim_outputs["T"] - params["T0"] + sim_outputs["delta_S"] + sim_outputs["delta_D"])) / (eps0 * averaged_rel_permitivity)
    if dEdx.ndim == 1:
        E_field = np.concatenate(([0], -np.cumsum(dEdx[::-1]) * params["Node_width"]))[::-1] #[V/nm]
        #E_field[-1] = 0
    else:
        E_field = np.concatenate((np.zeros(len(dEdx)).reshape((len(dEdx), 1)), -np.cumsum(dEdx[:, ::-1], axis=1) * params["Node_width"]), axis=1)[:, ::-1] #[V/nm]
        #E_field[:,-1] = 0
    return E_field
    
def delta_n(sim_outputs, params):
    """Calculate above-equilibrium electron density from N, n0"""
    return sim_outputs["N"] - params["N0"]

def delta_p(sim_outputs, params):
    """Calculate above-equilibrium hole density from P, p0"""
    return sim_outputs["P"] - params["P0"]

def delta_T(sim_outputs, params):
    """Calculate above-equilibrium triplet density from T, T0"""
    return sim_outputs["T"] - params["T0"]

def radiative_recombination(sim_outputs, params):
    """Calculate radiative recombination"""
    return params["B"] * (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"])

def nonradiative_recombination(sim_outputs, params):
    """Calculate nonradiative recombination using SRH model
       Assumes quasi steady state trap level occupation
      """
    return (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"]) / ((params["tau_N"] * sim_outputs["P"]) + (params["tau_P"] * sim_outputs["N"]))

def tau_diff(PL, dt):
    """
    Calculates particle lifetime from TRPL.

    Parameters
    ----------
    PL : 1D ndarray
        Time-resolved PL array.
    dt : float
        Time step size.

    Returns
    -------
    1D ndarray
        tau_diff.

    """
    try:
        ln_PL = np.log(PL)
    except Exception:
        print("Error: could not calculate tau_diff from non-positive PL values")
        return np.zeros(len(PL))
    dln_PLdt = np.zeros(len(ln_PL))
    dln_PLdt[0] = (ln_PL[1] - ln_PL[0]) / dt
    dln_PLdt[-1] = (ln_PL[-1] - ln_PL[-2]) / dt
    dln_PLdt[1:-1] = (np.roll(ln_PL, -1)[1:-1] - np.roll(ln_PL, 1)[1:-1]) / (2*dt)
    return -(dln_PLdt ** -1)

def prep_PL(rad_rec, i, j, need_extra_node, params, layer):
    """
    Calculates PL(x,t) given radiative recombination data plus propogation contributions.

    Parameters
    ----------
    radRec : 1D or 2D ndarray
        Radiative Recombination(x,t) values. These can be MAPI carriers or DBP doublets.
    i : int
        Leftmost node index to calculate for.
    j : int
        Rightmost node index to calculate for.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered.
        Most slices involving the index j should include j+1 too
    params : dict {"param name":float or 1D ndarray}
        Collection of parameters from metadata
    layer : str
        Whether to calculate for layer MAPI or layer DBP.
    Returns
    -------
    PL_base : 2D ndarray
        PL(x,t)

    """
    
    dx = params["Node_width"]
    
    lbound = to_pos(i, dx)
    if need_extra_node:
        ubound = to_pos(j+1, dx)
    else:
        ubound = to_pos(j, dx)
        
    distance = np.arange(lbound, ubound+dx, dx)
    
    if rad_rec.ndim == 2: # for integrals of partial thickness
        if need_extra_node:
            rad_rec = rad_rec[:,i:j+2]
        else:
            rad_rec = rad_rec[:,i:j+1]
            
        # If the nodes are not equally sized, the need_extra_node and 
        # to_pos mess up and make the distance array one too long. 
        # The user is discouraged from creating such nodes but if they do anyway,
        # Patch here and figure it out later.
        if len(distance) > len(rad_rec[0]):
            distance = distance[:len(rad_rec[0])]
            
    if layer == "MAPI":
        PL_base = (rad_rec)
        
    elif layer == "Rubrene":
        PL_base = rad_rec / params["tau_D"]
    
    return PL_base

def TTA(T, i, j, need_extra_node, params):
    """
    Calculates TTA rate(x,t) given T data.

    Parameters
    ----------
    T : 1D or 2D ndarray
        Triplet(x,t) values.
    i : int
        Leftmost node index to calculate for.
    j : int
        Rightmost node index to calculate for.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered.
        Most slices involving the index j should include j+1 too
    params : dict {"param name":float or 1D ndarray}
        Collection of parameters from metadata

    Returns
    -------
    PL_base : 2D ndarray
        PL(x,t)

    """
    
    dx = params["Node_width"]
    
    lbound = to_pos(i, dx)
    if need_extra_node:
        ubound = to_pos(j+1, dx)
    else:
        ubound = to_pos(j, dx)
        
    distance = np.arange(lbound, ubound+dx, dx)
    
    if T.ndim == 2: # for integrals of partial thickness
        if need_extra_node:
            T = T[:,i:j+2]
        else:
            T = T[:,i:j+1]
            
        # If the nodes are not equally sized, the need_extra_node and 
        # to_pos mess up and make the distance array one too long. 
        # The user is discouraged from creating such nodes but if they do anyway,
        # Patch here and figure it out later.
        if len(distance) > len(T[0]):
            distance = distance[:len(T[0])]
            
    dT = delta_T({"T":T}, params)
    TTA_base = (params["k_fusion"] * dT ** 2)

    return TTA_base

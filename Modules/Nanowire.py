# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
import numpy as np
from scipy import integrate as intg
from _helper_structs import Parameter, Output
from utils import u_read, to_index, to_array, to_pos, new_integrate
import tables
from _OneD_Model import OneD_Model

class Nanowire(OneD_Model):
    # A Nanowire object stores all information regarding the initial state being edited in the IC tab
    # And functions for managing other previously simulated nanowire data as they are loaded in
    def __init__(self):
        super().__init__()
        self.system_ID = "Nanowire"
        self.length_unit = "[nm]"
        self.time_unit = "[ns]"
        self.param_dict = {"Mu_N":Parameter(units="[cm^2 / V s]", is_edge=True), "Mu_P":Parameter(units="[cm^2 / V s]", is_edge=True), 
                            "N0":Parameter(units="[cm^-3]", is_edge=False), "P0":Parameter(units="[cm^-3]", is_edge=False), 
                            "B":Parameter(units="[cm^3 / s]", is_edge=False), "Tau_N":Parameter(units="[ns]", is_edge=False), 
                            "Tau_P":Parameter(units="[ns]", is_edge=False), "Sf":Parameter(units="[cm / s]", is_edge=False), 
                            "Sb":Parameter(units="[cm / s]", is_edge=False), "Temperature":Parameter(units="[K]", is_edge=True), 
                            "Rel-Permitivity":Parameter(units="", is_edge=True), "Ext_E-Field":Parameter(units="[V/um]", is_edge=True),
                            "Theta":Parameter(units="[cm^-1]", is_edge=False), "Alpha":Parameter(units="[cm^-1]", is_edge=False), 
                            "Delta":Parameter(units="", is_edge=False), "Frac-Emitted":Parameter(units="", is_edge=False),
                            "deltaN":Parameter(units="[cm^-3]", is_edge=False), "deltaP":Parameter(units="[cm^-3]", is_edge=False), 
                            "Ec":Parameter(units="[eV]", is_edge=True), "electron_affinity":Parameter(units="[eV]", is_edge=True)}
        

        self.param_count = len(self.param_dict)
        
        self.flags_dict = {"ignore_alpha":("Ignore Photon Recycle",0),
                           "symmetric_system":("Symmetric System",1),
                           "check_do_ss":("Steady State Input",0)}

        # List of all variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        self.simulation_outputs_dict = {"N":Output("N", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False,is_integrated=False, yscale='symlog', yfactors=(1e-4,1e1)), 
                                        "P":Output("P", units="[cm^-3]", xlabel="nm", xvar="position",is_edge=False,is_integrated=False, yscale='symlog', yfactors=(1e-4,1e1)),
                                        }
        
        # List of all variables calculated from those in simulation_outputs_dict
        self.calculated_outputs_dict = {"E_field":Output("Electric Field", units="[V/nm]", xlabel="nm", xvar="position",is_edge=True, calc_func=E_field, is_integrated=False),
                                        "deltaN":Output("delta_N", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False, calc_func=delta_n, is_integrated=False),
                                         "deltaP":Output("delta_P", units="[cm^-3]", xlabel="nm", xvar="position", is_edge=False, calc_func=delta_p, is_integrated=False),
                                         "RR":Output("Radiative Recombination", units="[cm^-3 s^-1]", xlabel="nm", xvar="position",is_edge=False, calc_func=radiative_recombination, is_integrated=False),
                                         "NRR":Output("Non-radiative Recombination", units="[cm^-3 s^-1]", xlabel="nm", xvar="position", is_edge=False, calc_func=nonradiative_recombination, is_integrated=False),
                                         "PL":Output("TRPL", units="[WIP]", xlabel="ns", xvar="time", is_edge=False, calc_func=new_integrate, is_integrated=True),
                                         "tau_diff":Output("-(dln(TRPL)/dt)^-1", units="[ns]", xlabel="ns", xvar="time", is_edge=False, calc_func=tau_diff, is_integrated=True, analysis_plotable=False)}
        
        self.outputs_dict = {**self.simulation_outputs_dict, **self.calculated_outputs_dict}
        
        self.simulation_outputs_count = len(self.simulation_outputs_dict)
        self.calculated_outputs_count = len(self.calculated_outputs_dict)
        self.total_outputs_count = self.simulation_outputs_count + self.calculated_outputs_count
        ## Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        self.convert_in_dict = {"Mu_N": ((1e7) ** 2) / (1e9), "Mu_P": ((1e7) ** 2) / (1e9), # [cm^2 / V s] to [nm^2 / V ns]
                                "N0": ((1e-7) ** 3), "P0": ((1e-7) ** 3),                   # [cm^-3] to [nm^-3]
                                "Thickness": 1, "dx": 1,
                                "B": ((1e7) ** 3) / (1e9),                                  # [cm^3 / s] to [nm^3 / ns]
                                "Tau_N": 1, "Tau_P": 1,                                     # [ns]
                                "Sf": (1e7) / (1e9), "Sb": (1e7) / (1e9),                   # [cm / s] to [nm / ns]
                                "Temperature": 1, "Rel-Permitivity": 1, 
                                "Ext_E-Field": 1e-3,                                        # [V/um] to [V/nm]
                                "Theta": 1e-7, "Alpha": 1e-7,                               # [cm^-1] to [nm^-1]
                                "Delta": 1, "Frac-Emitted": 1,
                                "deltaN": ((1e-7) ** 3), "deltaP": ((1e-7) ** 3),
                                "Ec": 1, "electron_affinity": 1,
                                "N": ((1e-7) ** 3), "P": ((1e-7) ** 3),                     # [cm^-3] to [nm^-3]
                                "E_field": 1, 
                                "tau_diff": 1}
        
        self.convert_in_dict["RR"] = self.convert_in_dict["B"] * self.convert_in_dict["N"] * self.convert_in_dict["P"]
        self.convert_in_dict["NRR"] = self.convert_in_dict["N"] / self.convert_in_dict["Tau_N"]
        self.convert_in_dict["PL"] = self.convert_in_dict["RR"] * self.convert_in_dict["Theta"]
        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
        self.convert_out_dict = {}
        for param in self.convert_in_dict:
            self.convert_out_dict[param] = self.convert_in_dict[param] ** -1

        return
    
    def calc_inits(self):
        """Calculate initial electron and hole density distribution"""
        init_N = (self.param_dict["N0"].value + self.param_dict["deltaN"].value) * self.convert_in_dict["N"]
        init_P = (self.param_dict["P0"].value + self.param_dict["deltaP"].value) * self.convert_in_dict["P"]
        # "Typecast" single values to uniform arrays
        if not isinstance(init_N, np.ndarray):
            init_N = np.ones(self.grid_x_nodes.__len__()) * init_N
            
        if not isinstance(init_P, np.ndarray):
            init_P = np.ones(self.grid_x_nodes.__len__()) * init_P
            
        
        return {"N":init_N, "P":init_P}
    
    def simulate(self, data_path, m, n, dt, params, flags, hmax_, init_conditions):
        """Calls ODEINT solver."""
        ode_nanowire(data_path, m, n, self.dx, dt, params,
                     not flags['ignore_alpha'].value(), 
                     flags['symmetric_system'].value(), 
                     flags['check_do_ss'].value(), hmax_, True,
                     init_conditions["N"], init_conditions["P"])
    
    def get_overview_analysis(self, params, tsteps, data_dirname, file_name_base):
        """Calculates at a selection of sample times: N, P, (total carrier densities)
           deltaN, deltaP, (above-equilibrium carrier densities)
           internal electric field due to differences in N, P,
           radiative recombination,
           non-radiative (SRH model) recombination,
           
           Integrates over nanowire length: PL due to radiative recombination, waveguiding, and carrier regeneration"""
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        data_dict = {}
        
        for raw_output_name in self.simulation_outputs_dict:
            data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, 
                                                 raw_output_name)
            data = []
            for tstep in tsteps:
                data.append(u_read(data_filename, t0=tstep, single_tstep=True))
            
            data_dict[raw_output_name] = np.array(data)
            
            
        for calculated_output_name, output_obj in self.calculated_outputs_dict.items():
            if not output_obj.is_integrated:
                data_dict[calculated_output_name] = output_obj.calc_func(data_dict, params)
                
                
        with tables.open_file(data_dirname + "\\" + file_name_base + "-n.h5", mode='r') as ifstream_N, \
            tables.open_file(data_dirname + "\\" + file_name_base + "-p.h5", mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data)
            temp_P = np.array(ifstream_P.root.data)
        temp_RR = radiative_recombination({"N":temp_N, "P":temp_P}, params)
        PL_base = prep_PL(temp_RR, 0, to_index(params["Total_length"], 
                                               params["Node_width"], 
                                               params["Total_length"]), False, params)
        data_dict["PL"] = self.calculated_outputs_dict["PL"].calc_func(PL_base, 0, params["Total_length"], 
                                                                       params["Node_width"], params["Total_length"], 
                                                                       False)
        data_dict["tau_diff"] = self.calculated_outputs_dict["tau_diff"].calc_func(data_dict["PL"], params["dt"])
        
        for data in data_dict:
            data_dict[data] *= self.convert_out_dict[data]
        
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, for_integrate=False, 
                     i=0, j=0, nen=False, extra_data = None):
        """ Provides deltaN, deltaP, electric field, recombination, 
            and spatial PL values on demand.
        """
        # For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        data = None
        if (datatype in self.simulation_outputs_dict):
            data = sim_data[datatype]
        
        else:
            if (datatype == "deltaN"):
                data = delta_n(sim_data, params)
                
            elif (datatype == "deltaP"):
                data = delta_p(sim_data, params)
                
            elif (datatype == "RR"):
                data = radiative_recombination(sim_data, params)

            elif (datatype == "NRR"):
                data = nonradiative_recombination(sim_data, params)
                
            elif (datatype == "E_field"):
                data = E_field(sim_data, params)

            elif (datatype == "PL"):
    
                if for_integrate:
                    rad_rec = radiative_recombination(extra_data, params)
                    data = prep_PL(rad_rec, i, j, nen, params)
                else:
                    rad_rec = radiative_recombination(sim_data, params)
                    data = prep_PL(rad_rec, 0, len(rad_rec), need_extra_node=False, 
                                   params=params).flatten()
            else:
                raise ValueError
                
        return data
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        """ Set deltaN and deltaP of outgoing regenerated IC file."""
        param_dict["deltaN"] = (sim_data["N"] - param_dict["N0"]) if include_flags['N'] else np.zeros(grid_x.__len__())
                    
        param_dict["deltaP"] = (sim_data["P"] - param_dict["P0"]) if include_flags['P'] else np.zeros(grid_x.__len__())

        return
    
def gen_weight_distribution(m, dx, alphaCof=0, thetaCof=0, delta_frac=1, 
                            fracEmitted=0, symmetric=True):
    """
    Distance-dependent alpha weighting matrix for nanowire regeneration term

    Parameters
    ----------
    m : int
        Number of space nodes.
    dx : float
        Space node width.
    alphaCof : float, optional
        Regeneration coefficient. The default is 0.
    thetaCof : float, optional
        Photon propagation coefficient. The default is 0.
    delta_frac : float, optional
        Proportion of excitons affected by regeneration only. The default is 1.
    fracEmitted : float, optional
        Proportion of photons that escape the nanowire (and thus do not propogate). The default is 0.
    symmetric : bool, optional
        Whether contributions from symmetric half of nanowire should be considered. The default is True.

    Returns
    -------
    2D ndarray
        Weight matrix of size (m,m).

    """
    distance = np.arange(0, m*dx, dx)
    distance_matrix = np.zeros((m, m))
    lf_distance_matrix = np.zeros((m, m)) # Account for "other half" of a symmetric system

    # Each row in distance_matrix represents the weight function centered around a different position
    # Element [i,j] is the proportion of node j's value that contributes to node i
    for i in range(0,m):
        distance_matrix[i] = np.concatenate((np.flip(distance[0:i+1], 0), distance[1:m - i]))
        if symmetric: 
            lf_distance_matrix[i] = distance + ((i+1) * dx)
    
    weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
    if symmetric: 
        weight += np.exp(-(alphaCof + thetaCof) * lf_distance_matrix)
    return alphaCof * 0.5 * (1 - fracEmitted) * delta_frac * weight

def dydt2(t, y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, 
          eps, eps0, q, q_C, kB, recycle_photons=True, do_ss=False, 
          alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, 
          combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_N=0, init_P=0):
    """Derivative function for drift-diffusion-decay carrier model."""
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((m+1))
    Jp = np.zeros((m+1))

    # These are calculated at node centers, of which there are m
    # dE/dt, dn/dt, and dp/dt are also node center values
    dJz = np.zeros((m))
    rad_rec = np.zeros((m))
    non_rad_rec = np.zeros((m))
    G_array = np.zeros((m))

    N = y[0:m]
    P = y[m:2*(m)]
    E_field = y[2*(m):]
    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    
    ## Do boundary conditions of Jn, Jp
    # FIXME: Calculate N, P at boundaries?
    Sft = Sf * (N[0] * P[0] - n0[0] * p0[0]) / (N[0] + P[0])
    Sbt = Sb * (N[m-1] * P[m-1] - n0[m-1] * p0[m-1]) / (N[m-1] + P[m-1])
    Jn[0] = Sft
    Jn[m] = -Sbt
    Jp[0] = -Sft
    Jp[m] = Sbt

    ## Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension, 
    # Jn(t) ~ N(t) * E_field(t) + (dN/dt)
    # np.roll(y,m) shifts the values of array y by m places, allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx) over entire array y
    Jn[1:-1] = (-mu_n[1:-1] * (N_edges) * (q * (E_field[1:-1] + E_field_ext[1:-1]) + dChidz[1:-1]) 
                + (mu_n[1:-1]*kB*T[1:-1]) * ((np.roll(N,-1)[:-1] - N[:-1]) / (dx)))

    ## Changed sign
    Jp[1:-1] = (-mu_p[1:-1] * (P_edges) * (q * (E_field[1:-1] + E_field_ext[1:-1]) + dChidz[1:-1] + dEcdz[1:-1]) 
                -(mu_p[1:-1]*kB*T[1:-1]) * ((np.roll(P, -1)[:-1] - P[:-1]) / (dx)))

        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (eps * eps0))
    
    ## Calculate recombination (consumption) terms
    rad_rec = B * (N * P - n0 * p0)
    non_rad_rec = (N * P - n0 * p0) / ((tauN * P) + (tauP * N))
        
    ## Calculate generation term from photon recycling, if photon recycling is being considered
    if recycle_photons:
        G_array = intg.trapz(rad_rec * combined_weight, dx=dx, axis=1) \
                  + (1 - fracEmitted) * 0.5 * alphaCof * delta_frac * rad_rec
    else:
        G_array = 0
    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dx)

    ## N(t) = N(t-1) + dt * (dN/dt)
    #N_new = np.maximum(N_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: 
        dNdt += init_N

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dx)

    ## P(t) = P(t-1) + dt * (dP/dt)
    #P_new = np.maximum(P_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: 
        dPdt += init_P

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt
    
def ode_nanowire(data_path_name, m, n, dx, dt, params, recycle_photons=True, 
                 symmetric=True, do_ss=False, hmax_=0, write_output=True, 
                 init_N=0, init_P=0, init_E_field=0):
    """
    Master function for Nanowire module simulation.
    Problem statement:
    Create a discretized, time and space dependent solution (N(x,t) and P(x,t)) of the carrier model with m space steps and n time steps
    Space step size is dx, time step is dt
    Initial conditions: init_N, init_P, init_E_field
    Optional photon recycle term

    Parameters
    ----------
    data_path_name : str
        Output file location.
    m : int
        Number of space nodes.
    n : int
        Number of time steps.
    dx : float
        Space node width.
    dt : float
        Time step size.
    params : dict {"str":float or 1D array}
        Collection of parameter values
    recycle_photons : bool, optional
        Whether carrier regeneration due to photons is considered. The default is True.
    symmetric : bool, optional
        Whether to consider the nanowire as having a symmetrical half with virtual nodes 0 to -m. The default is True.
    do_ss : bool, optional
        Whether to inject the initial conditions at every time step, creating a nonzero steady state situation. The default is False.
    hmax_ : float, optional
        Maximum internal step size to be taken by ODEINT. The default is 0.
    write_output : bool, optional
        Whether to write output files. TEDs always does this but other applications reusing this function might not. The default is True.
    init_N : 1D ndarray, optional
        Initial excited electron distribution. The default is 0.
    init_P : 1D ndarray, optional
        Initial hole distribution. The default is 0.
    init_E_field : 1D ndarray, optional
        Initial electric field. The default is 0.

    Returns
    -------
    None
        TEDs does not do anything with the return value. Other applications might find this useful however.
    """
    
    ## Set data type of array files
    atom = tables.Float64Atom()

    ## Unpack params; typecast non-array params to arrays if needed
    Sf = params["Sf"]
    Sb = params["Sb"]
    mu_n = to_array(params["Mu_N"], m, True)
    mu_p = to_array(params["Mu_P"], m, True)
    T = to_array(params["Temperature"], m, True)
    n0 = to_array(params["N0"], m, False)
    p0 = to_array(params["P0"], m, False)
    tauN = to_array(params["Tau_N"], m, False)
    tauP = to_array(params["Tau_P"], m, False)
    B = to_array(params["B"], m, False)
    eps = to_array(params["Rel-Permitivity"], m, True)
    E_field_ext = to_array(params["Ext_E-Field"], m, True)
    alphaCof = to_array(params["Alpha"], m, False) if recycle_photons else np.zeros(m)
    thetaCof = to_array(params["Theta"], m, False)
    delta_frac = to_array(params["Delta"], m, False)
    fracEmitted = to_array(params["Frac-Emitted"], m, False)
    init_Ec = to_array(params["Ec"], m, True)
    init_Chi = to_array(params["electron_affinity"], m, True)
           
    ## Define constants
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q = 1.0 # [e]
    q_C = 1.602e-19 # [C]
    kB = 8.61773e-5  # [eV / K]
    
    ## Package initial condition
    # An unfortunate workaround - create temporary dictionaries out of necessary values to match the call signature of E_field()
    init_E_field = E_field({"N":init_N, "P":init_P}, 
                           {"Rel-Permitivity":eps, "N0":n0, "P0":p0, "Node_width":dx})
    #init_E_field = np.zeros(m+1)
    
    init_condition = np.concatenate([init_N, init_P, init_E_field], axis=None)

    if do_ss:
        init_N_copy = init_N
        init_P_copy = init_P

    else:
        init_N_copy = 0
        init_P_copy = 0

    ## Generate a weight distribution needed for photon recycle term if photon recycle is being considered
    if recycle_photons:
        combined_weight = gen_weight_distribution(m, dx, alphaCof, thetaCof, 
                                                  delta_frac, fracEmitted, symmetric)
    else:
        combined_weight = np.zeros((m, m))

    ## Generate space derivative of Ec and Chi
    # Note that for these two quantities the derivatives at node edges are being calculated by values at node edges
    # This is not recommended for N, P (their derivatives are calculated from node centers) because N, P are used to model discrete chunks of nanowire over time,
    # but it is okay to use Ec, Chi because these are invariant with time.

    dEcdz = np.zeros(m+1)
    dChidz = np.zeros(m+1)

    dEcdz[1:-1] = (np.roll(init_Ec, -1)[1:-1] - np.roll(init_Ec, 1)[1:-1]) / (2 * dx)
    dEcdz[0] = (init_Ec[1] - init_Ec[0]) / dx
    dEcdz[m] = (init_Ec[m] - init_Ec[m-1]) / dx

    dChidz[1:-1] = (np.roll(init_Chi, -1)[1:-1] - np.roll(init_Chi, 1)[1:-1]) / (2 * dx)
    dChidz[0] = (init_Chi[1] - init_Chi[0]) / dx
    dChidz[m] = (init_Chi[m] - init_Chi[m-1]) / dx


    ## Do n time steps
    tSteps = np.linspace(0, n*dt, n+1)
    data, error_data = intg.odeint(dydt2, init_condition, tSteps, 
                                   args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, 
                                         tauN, tauP, B, eps, eps0, q, q_C, kB, 
                                         recycle_photons, do_ss, alphaCof, thetaCof, 
                                         delta_frac, fracEmitted, combined_weight, 
                                         E_field_ext, dEcdz, dChidz, init_N_copy, 
                                         init_P_copy),
                                   tfirst=True, full_output=True, hmax=hmax_)
        
    if (data[1:, 0:2*m] < 0).any():
        h = np.geomspace(2**2, 2**-6, 9)
        for hmax in h:
            print("Simulation is not converging well, retrying with hmax={}".format(hmax))
            data, error_data = intg.odeint(dydt2, init_condition, tSteps, 
                                           args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, 
                                                 tauN, tauP, B, eps, eps0, q, q_C, kB, 
                                                 recycle_photons, do_ss, alphaCof, thetaCof, 
                                                 delta_frac, fracEmitted, combined_weight, 
                                                 E_field_ext, dEcdz, dChidz, init_N_copy, 
                                                 init_P_copy),
                                           tfirst=True, full_output=True, hmax=hmax)
                
            if not (data[1:, 0:2*m] < 0).any():
                break
            
    if write_output:
        ## Prep output files
        with tables.open_file(data_path_name + "-N.h5", mode='a') as ofstream_N, \
            tables.open_file(data_path_name + "-P.h5", mode='a') as ofstream_P:
            #tables.open_file(data_path_name + "-E_field.h5", mode='a') as ofstream_E_field:
            array_N = ofstream_N.root.data
            array_P = ofstream_P.root.data
            #array_E_field = ofstream_E_field.root.data
            array_N.append(data[1:,0:m])
            array_P.append(data[1:,m:2*(m)])
            #array_E_field.append(data[1:,2*(m):])

        return error_data

    else:
        array_N = data[:,0:m]
        array_P = data[:,m:2*(m)]

        return array_N, array_P, error_data
    
def E_field(sim_outputs, params):
    """Calculate electric field from N, P"""
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q_C = 1.602e-19 # [C per carrier]
    if isinstance(params["Rel-Permitivity"], np.ndarray):
        averaged_rel_permitivity = (params["Rel-Permitivity"][:-1] + np.roll(params["Rel-Permitivity"], -1)[:-1]) / 2
    else:
        averaged_rel_permitivity = params["Rel-Permitivity"]
    
    dEdx = q_C * (delta_p(sim_outputs, params) - delta_n(sim_outputs, params)) / (eps0 * averaged_rel_permitivity)
    if dEdx.ndim == 1:
        E_field = np.concatenate(([0], np.cumsum(dEdx) * params["Node_width"])) #[V/nm]
        E_field[-1] = 0
    else:
        E_field = np.concatenate((np.zeros(len(dEdx)).reshape((len(dEdx), 1)), np.cumsum(dEdx, axis=1) * params["Node_width"]), axis=1) #[V/nm]
        E_field[:,-1] = 0
    return E_field
    
def delta_n(sim_outputs, params):
    """Calculate above-equilibrium electron density from N, n0"""
    return sim_outputs["N"] - params["N0"]

def delta_p(sim_outputs, params):
    """Calculate above-equilibrium hole density from P, p0"""
    return sim_outputs["P"] - params["P0"]

def radiative_recombination(sim_outputs, params):
    """Calculate radiative recombination"""
    return params["B"] * (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"])

def nonradiative_recombination(sim_outputs, params):
    """Calculate nonradiative recombination using SRH model
       Assumes quasi steady state trap level occupation
      """
    return (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"]) / ((params["Tau_N"] * sim_outputs["P"]) + (params["Tau_P"] * sim_outputs["N"]))

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
    dln_PLdt = np.zeros(ln_PL.__len__())
    dln_PLdt[0] = (ln_PL[1] - ln_PL[0]) / dt
    dln_PLdt[-1] = (ln_PL[-1] - ln_PL[-2]) / dt
    dln_PLdt[1:-1] = (np.roll(ln_PL, -1)[1:-1] - np.roll(ln_PL, 1)[1:-1]) / (2*dt)
    return -(dln_PLdt ** -1)

def PL_weight_distribution(m, dx, total_length, i, j, alpha, theta, delta, 
                           frac_emitted, need_extra_node, symmetric=True):
    """
    Distance-dependent alpha+theta weighting for PL propagation
    
    Parameters
    ----------
    total_length : float
        Length of nanowire.
    i : int
        Index of leftmost node being integrated over
    j : int
        Index of rightmost node being integrated over
    need_extra_node : bool
        Whether the 'j+1'th node should be included. 
        This is a correction between the discrete nodes and the actual bounds of integration.
    See gen_weight_distribution() for more details
    Returns
    -------
    2D ndarray
        Weighting matrix of size [(j-i+1),m] or [(j-i+2),m].

    """
    distance = np.arange(0, total_length, dx)

    # Make room for the extra node if needed
    r = j + 2 if need_extra_node else j+1

    distance_matrix = np.zeros((r - i, m))
    lf_distance_matrix = np.zeros((r - i, m))

    # Each row in weight will represent the weight function centered around a different position
    # Additional lf_weight counted in if wire is symmetric
    # Element [u,v] is the proportion of node v that contributes to node u.
    for n in range(i, r):
        distance_matrix[n - i] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:m - n]))
        lf_distance_matrix[n - i] = distance + ((n+1) * dx)
    
    weight = np.exp(-(alpha + theta) * distance_matrix)
    weight2 = np.exp(-(theta) * distance_matrix)
    
    if symmetric:
        weight += np.exp(-(alpha + theta) * lf_distance_matrix) 
        weight2 += np.exp(-(theta) * lf_distance_matrix)
        
    return (1 - frac_emitted) * 0.5 * theta * (delta * weight + (1 - delta) * weight2)

def prep_PL(radRec, i, j, need_extra_node, params):
    """
    Calculates PL(x,t) given radiative recombination data plus propogation contributions.

    Parameters
    ----------
    radRec : 1D or 2D ndarray
        Radiative Recombination(x,t) values.
    i : int
        Leftmost node to calculate for.
    j : int
        Rightmost node index to calculate for.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered
    params : dict {"param name":float or 1D ndarray}
        Collection of parameters from metadata

    Returns
    -------
    PL_base : 2D ndarray
        PL(x,t)

    """
    
    frac_emitted = params["Frac-Emitted"]
    alpha = 0 if params["ignore_alpha"] else params["Alpha"]
    theta = params["Theta"]
    delta = params["Delta"]
    dx = params["Node_width"]
    total_length = params["Total_length"]
    m = int(total_length / dx)
    
    if np.ndim(radRec) == 1:
        radRec = radRec[None]
            
    if need_extra_node:
        temp_RR = radRec[:, i:j+2]
    else:
        temp_RR = radRec[:, i:j+1]
    PL_base = frac_emitted * temp_RR
    
    combined_weight = PL_weight_distribution(m, dx, total_length, i, j, alpha, 
                                             theta, delta, frac_emitted, need_extra_node, 
                                             params["symmetric_system"])

    for p in range(len(PL_base[0])):
        PL_base[:,p] += intg.trapz(combined_weight[p] * radRec, dx=dx, axis=1).T \
                        + radRec[:,i+p] * theta * (1-frac_emitted) * 0.5
    
    return PL_base
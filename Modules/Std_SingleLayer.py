# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
import numpy as np
import os
from scipy import integrate as intg
from helper_structs import Parameter, Output, Layer
from utils import u_read, to_index, to_array, to_pos, new_integrate
import tables
from _OneD_Model import OneD_Model

class Std_SingleLayer(OneD_Model):
    # A Nanowire object stores all information regarding the initial state being edited in the IC tab
    # And functions for managing other previously simulated nanowire data as they are loaded in
    def __init__(self):
        super().__init__()
        self.system_ID = "OneLayer"
        self.time_unit = "[ns]"
        params = {"mu_N":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
                  "mu_P":Parameter(units="[cm^2 / V s]", is_edge=True, valid_range=(0,np.inf)), 
                  "N0":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
                  "P0":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
                  "B":Parameter(units="[cm^3 / s]", is_edge=False, valid_range=(0,np.inf)), 
                  "tau_N":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
                  "tau_P":Parameter(units="[ns]", is_edge=False, valid_range=(0,np.inf)), 
                  "Sf":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
                  "Sb":Parameter(units="[cm / s]", is_edge=False, is_space_dependent=False, valid_range=(0,np.inf)), 
                  "temperature":Parameter(units="[K]", is_edge=True, valid_range=(0,np.inf)), 
                  "rel_permitivity":Parameter(units="", is_edge=True, valid_range=(0,np.inf)), 
                  "Ext_E-Field":Parameter(units="[V/um]", is_edge=True),
                  "back_reflectivity":Parameter(units="", is_edge=False, is_space_dependent=False, valid_range=(0,1)), 
                  "alpha":Parameter(units="[cm^-1]", is_edge=False, valid_range=(0,np.inf)), 
                  "delta":Parameter(units="", is_edge=False, valid_range=(0,1)), 
                  "frac_emitted":Parameter(units="", is_edge=False, is_space_dependent=False, valid_range=(0,1)),
                  "delta_N":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
                  "delta_P":Parameter(units="[carr / cm^3]", is_edge=False, valid_range=(0,np.inf)), 
                  "Ec":Parameter(units="[eV]", is_edge=True), 
                  "electron_affinity":Parameter(units="[eV]", is_edge=True)}
                
        self.flags_dict = {"ignore_recycle":("Ignore Photon Recycle",1, 0),
                           #"symmetric_system":("Symmetric System",0, 0),
                           "check_do_ss":("Steady State Input",1, 0)}

        # List of all variables active during the finite difference simulating        
        # calc_inits() must return values for each of these or an error will be raised!
        simulation_outputs = {"N":Output("N", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="OneLayer", yscale='symlog', yfactors=(1e-4,1e1)), 
                              "P":Output("P", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position",is_edge=False, layer="OneLayer", yscale='symlog', yfactors=(1e-4,1e1)),
                             }
        
        # List of all variables calculated from those in simulation_outputs_dict
        calculated_outputs = {"E_field":Output("Electric Field", units="[V/nm]", integrated_units="[V]", xlabel="nm", xvar="position",is_edge=True, layer="OneLayer"),
                             "delta_N":Output("delta_N", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="OneLayer"),
                             "delta_P":Output("delta_P", units="[carr / cm^3]", integrated_units="[carr / cm^2]", xlabel="nm", xvar="position", is_edge=False, layer="OneLayer"),
                             "RR":Output("Radiative Recombination", units="[carr / cm^3 s]", integrated_units="[carr / cm^2 s]", xlabel="nm", xvar="position",is_edge=False, layer="OneLayer"),
                             "NRR":Output("Non-radiative Recombination", units="[carr / cm^3 s]", integrated_units="[carr / cm^2 s]", xlabel="nm", xvar="position", is_edge=False, layer="OneLayer"),
                             "PL":Output("TRPL", units="[phot / cm^3 s]", integrated_units="[phot / cm^2 s]", xlabel="ns", xvar="time", is_edge=False, layer="OneLayer"),
                             "tau_diff":Output("tau_diff", units="[ns]", xlabel="ns", xvar="time", is_edge=False, layer="OneLayer", analysis_plotable=False)}
        
        ## Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        convert_in = {"mu_N": ((1e7) ** 2) / (1e9), "mu_P": ((1e7) ** 2) / (1e9), # [cm^2 / V s] to [nm^2 / V ns]
                      "N0": ((1e-7) ** 3), "P0": ((1e-7) ** 3),                   # [cm^-3] to [nm^-3]
                      "B": ((1e7) ** 3) / (1e9),                                  # [cm^3 / s] to [nm^3 / ns]
                      "tau_N": 1, "tau_P": 1,                                     # [ns]
                      "Sf": (1e7) / (1e9), "Sb": (1e7) / (1e9),                   # [cm / s] to [nm / ns]
                      "temperature": 1, "rel_permitivity": 1, 
                      "Ext_E-Field": 1e-3,                                        # [V/um] to [V/nm]
                      "alpha": 1e-7,                               # [cm^-1] to [nm^-1]
                      "delta": 1, "frac_emitted": 1,
                      "back_reflectivity": 1,
                      "delta_N": ((1e-7) ** 3), "delta_P": ((1e-7) ** 3),
                      "Ec": 1, "electron_affinity": 1,
                      "N": ((1e-7) ** 3), "P": ((1e-7) ** 3),                     # [cm^-3] to [nm^-3]
                      "E_field": 1, 
                      "tau_diff": 1}
        
        convert_in["RR"] = convert_in["B"] * convert_in["N"] * convert_in["P"] # [cm^-3 s^-1] to [nm^-3 ns^-1]
        convert_in["NRR"] = convert_in["N"] * 1e-9 # [cm^-3 s^-1] to [nm^-3 ns^-1]
        convert_in["PL"] = convert_in["RR"]
        
        iconvert_in = {"N":1e7, "P":1e7, "delta_N":1e7, "delta_P":1e7, # cm to nm
                       "E_field":1, # nm to nm
                       "RR": 1e7, "NRR": 1e7, "PL": 1e7}

        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
            
        self.layers = {"OneLayer":Layer(params, simulation_outputs, calculated_outputs,
                                        "[nm]", convert_in, iconvert_in),
                       }

        return
    
    def calc_inits(self):
        """Calculate initial electron and hole density distribution"""
        one_layer = self.layers["OneLayer"]
        init_N = (one_layer.params["N0"].value + one_layer.params["delta_N"].value) * one_layer.convert_in["N"]
        init_P = (one_layer.params["P0"].value + one_layer.params["delta_P"].value) * one_layer.convert_in["P"]
        # "Typecast" single values to uniform arrays
        if not isinstance(init_N, np.ndarray):
            init_N = np.ones(len(one_layer.grid_x_nodes)) * init_N
            
        if not isinstance(init_P, np.ndarray):
            init_P = np.ones(len(one_layer.grid_x_nodes)) * init_P
            
        
        return {"N":init_N, "P":init_P}
    
    def simulate(self, data_path, m, n, dt, flags, hmax_, init_conditions):
        """Calls ODE solver."""
        one_layer = self.layers["OneLayer"]
        for param_name, param in one_layer.params.items():
            param.value *= one_layer.convert_in[param_name]

        ode_onelayer(data_path, m["OneLayer"], n, one_layer.dx, dt, one_layer.params,
                     not flags['ignore_recycle'].value(), 
                     flags['check_do_ss'].value(), hmax_, True,
                     init_conditions["N"], init_conditions["P"])
    
    def get_overview_analysis(self, params, flags, total_time, dt, tsteps, data_dirname, file_name_base):
        """Calculates at a selection of sample times: N, P, (total carrier densities)
           delta_N, delta_P, (above-equilibrium carrier densities)
           internal electric field due to differences in N, P,
           radiative recombination,
           non-radiative (SRH model) recombination,
           
           Integrates over nanowire length: PL due to radiative recombination"""
        # Must return: a dict indexed by output names in self.output_dict containing 1- or 2D numpy arrays
        one_layer = self.layers["OneLayer"]
        params = params["OneLayer"]
        total_length = params["Total_length"]
        dx = params["Node_width"]
        data_dict = {"OneLayer":{}}
        
        for raw_output_name in one_layer.s_outputs:
            data_filename = "{}/{}-{}.h5".format(data_dirname, file_name_base, 
                                                 raw_output_name)
            data = []
            for tstep in tsteps:
                data.append(u_read(data_filename, t0=tstep, single_tstep=True))
            
            data_dict["OneLayer"][raw_output_name] = np.array(data)
                    
        data_dict["OneLayer"]["E_field"] = E_field(data_dict["OneLayer"], params)
        data_dict["OneLayer"]["delta_N"] = delta_n(data_dict["OneLayer"], params)
        data_dict["OneLayer"]["delta_P"] = delta_p(data_dict["OneLayer"], params)
        data_dict["OneLayer"]["RR"] = radiative_recombination(data_dict["OneLayer"], params)
        data_dict["OneLayer"]["NRR"] = nonradiative_recombination(data_dict["OneLayer"], params)
                
        with tables.open_file(os.path.join(data_dirname, file_name_base + "-N.h5"), mode='r') as ifstream_N, \
            tables.open_file(os.path.join(data_dirname, file_name_base + "-P.h5"), mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data)
            temp_P = np.array(ifstream_P.root.data)
        temp_RR = radiative_recombination({"N":temp_N, "P":temp_P}, params)
        PL_base = prep_PL(temp_RR, 0, to_index(total_length, dx, total_length), 
                          False, params, flags["ignore_recycle"])
        data_dict["OneLayer"]["PL"] = new_integrate(PL_base, 0, total_length, 
                                                    dx, total_length, False)
        data_dict["OneLayer"]["tau_diff"] = tau_diff(data_dict["OneLayer"]["PL"], dt)
        
        for data in data_dict["OneLayer"]:
            data_dict["OneLayer"][data] *= one_layer.convert_out[data]
            
        data_dict["OneLayer"]["PL"] *= one_layer.iconvert_out["PL"]
        
        return data_dict
    
    def prep_dataset(self, datatype, sim_data, params, flags, for_integrate=False, 
                     i=0, j=0, nen=False, extra_data = None):
        """ Provides delta_N, delta_P, electric field, recombination, 
            and spatial PL values on demand.
        """
        # For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        one_layer = self.layers["OneLayer"]
        params = params["OneLayer"]
        sim_data = sim_data["OneLayer"]
        ignore_recycle = flags["ignore_recycle"]
        data = None
        if (datatype in one_layer.s_outputs):
            data = sim_data[datatype]
        
        else:
            if (datatype == "delta_N"):
                data = delta_n(sim_data, params)
                
            elif (datatype == "delta_P"):
                data = delta_p(sim_data, params)
                
            elif (datatype == "RR"):
                data = radiative_recombination(sim_data, params)

            elif (datatype == "NRR"):
                data = nonradiative_recombination(sim_data, params)
                
            elif (datatype == "E_field"):
                data = E_field(sim_data, params)

            elif (datatype == "PL"):
    
                if for_integrate:
                    rad_rec = radiative_recombination(extra_data["OneLayer"], params)
                    data = prep_PL(rad_rec, i, j, nen, params, ignore_recycle)
                else:
                    rad_rec = radiative_recombination(sim_data, params)
                    data = prep_PL(rad_rec, 0, len(rad_rec)-1, False, 
                                   params, ignore_recycle).flatten()
            else:
                raise ValueError
                
        return data
    
    def get_timeseries(self, pathname, datatype, parent_data, total_time, dt, params, flags):
        """ Calculates supplemental data - the effective lifetime - for integrated PL"""
        if datatype == "PL":
            return [("tau_diff", tau_diff(parent_data, dt))]
        
        else:
            return
    
    def get_IC_carry(self, sim_data, param_dict, include_flags, grid_x):
        """ Set delta_N and delta_P of outgoing regenerated IC file."""
        param_dict = param_dict["OneLayer"]
        sim_data = sim_data["OneLayer"]
        include_flags = include_flags["OneLayer"]
        param_dict["delta_N"] = (sim_data["N"] - param_dict["N0"]) if include_flags['N'] else np.zeros(len(grid_x))
                    
        param_dict["delta_P"] = (sim_data["P"] - param_dict["P0"]) if include_flags['P'] else np.zeros(len(grid_x))

        return
    
def gen_weight_distribution(m, dx, alpha=0, delta_frac=1, 
                            back_refl_frac=1, frac_emitted=0):
    """
    Distance-dependent weighting matrix for one-layer carrier regeneration term

    Parameters
    ----------
    m : int
        Number of space nodes.
    dx : float
        Space node width.
    alpha : float, optional
        Absorption coefficient. The default is 0.
    back_refl_frac : float, optional
        Proportion of photons which reflect off the back surface.
        The default is 1.
    delta_frac : float, optional
        Fraction of radiative recombination which overlaps 
        with material absorption spectrum. 
        The default is 1.
    frac_emitted : float, optional
        Proportion of photons that escape the system at surface 
        (and thus do not reflect back inward). 
        The default is 0.
    
    Returns
    -------
    2D ndarray
        Weight matrix of size (m,m).

    """
    distance = np.arange(0, m*dx, dx)
    direct_distance_matrix = np.zeros((m, m))
    front_refl_distance_matrix = np.zeros((m, m))
    back_refl_distance_matrix = np.zeros((m, m))

    # Each row in distance_matrix represents the weight function centered around a different position
    # Element [i,j] is the proportion of node j's value that contributes to node i
    for i in range(0,m):
        direct_distance_matrix[i] = np.concatenate((np.flip(distance[0:i+1], 0), distance[1:m - i]))
        front_refl_distance_matrix[i] = distance + ((i+1) * dx)
        back_refl_distance_matrix[i] = 2 * m * dx - (distance + ((i+1)*dx))
        
    direct_weight = np.exp((-alpha) * direct_distance_matrix)
    
    front_refl_weight = ((1 - frac_emitted)
                        * np.exp((-alpha) * front_refl_distance_matrix)
                        )
    back_refl_weight = (back_refl_frac
                       * np.exp((-alpha) * back_refl_distance_matrix)
                       )
    return (alpha * delta_frac / 2 * 
            (direct_weight + front_refl_weight + back_refl_weight)
           )

def dydt2(t, y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, 
          eps, eps0, q, q_C, kB, recycle_photons=True, do_ss=False, 
          alpha=0, back_refl_frac=1, delta_frac=1, frac_emitted=0, 
          combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_dN=0, init_dP=0):
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
        G_array = (intg.trapz(rad_rec * combined_weight, dx=dx, axis=1) 
                   + rad_rec * (0.5 * alpha * delta_frac)
                   * (1
                     - ((1 - frac_emitted) * np.exp(-2 * alpha * np.arange(0, m*dx, dx)))
                     - (back_refl_frac * np.exp(-2 * alpha * (m*dx - np.arange(0, m*dx, dx))))
                     )
                  )
    else:
        G_array = 0
    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dx)

    ## N(t) = N(t-1) + dt * (dN/dt)
    #N_new = np.maximum(N_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: 
        dNdt += init_dN

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dx)

    ## P(t) = P(t-1) + dt * (dP/dt)
    #P_new = np.maximum(P_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: 
        dPdt += init_dP

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt
    
def ode_onelayer(data_path_name, m, n, dx, dt, params, recycle_photons=True, 
                 do_ss=False, hmax_=0, write_output=True, 
                 init_N=0, init_P=0):
    """
    Master function for Onelayer module simulation.
    Problem statement:
    Create a discretized, time and space dependent solution (N(x,t) and P(x,t)) of the carrier model with m space steps and n time steps
    Space step size is dx, time step is dt
    Initial conditions: init_N, init_P
    Optional carrier recycle term

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
    params : dict {"str":Parameter}
        Collection of parameter objects
    recycle_photons : bool, optional
        Whether carrier regeneration due to photons is considered. The default is True.
    do_ss : bool, optional
        Whether to inject the initial conditions at every time step, creating a nonzero steady state situation. The default is False.
    hmax_ : float, optional
        Maximum internal step size to be taken by ODE solver. The default is 0.
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
    Sf = params["Sf"].value
    Sb = params["Sb"].value
    mu_n = to_array(params["mu_N"].value, m, True)
    mu_p = to_array(params["mu_P"].value, m, True)
    T = to_array(params["temperature"].value, m, True)
    n0 = to_array(params["N0"].value, m, False)
    p0 = to_array(params["P0"].value, m, False)
    tauN = to_array(params["tau_N"].value, m, False)
    tauP = to_array(params["tau_P"].value, m, False)
    B = to_array(params["B"].value, m, False)
    eps = to_array(params["rel_permitivity"].value, m, True)
    E_field_ext = to_array(params["Ext_E-Field"].value, m, True)
    alpha = to_array(params["alpha"].value, m, False)
    back_refl_frac = params["back_reflectivity"].value
    delta_frac = to_array(params["delta"].value, m, False) if recycle_photons else np.zeros(m)
    frac_emitted = to_array(params["frac_emitted"].value, m, False)
    init_Ec = to_array(params["Ec"].value, m, True)
    init_Chi = to_array(params["electron_affinity"].value, m, True)
           
    ## Define constants
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q = 1.0 # [e]
    q_C = 1.602e-19 # [C]
    kB = 8.61773e-5  # [eV / K]
    
    ## Package initial condition
    # An unfortunate workaround - create temporary dictionaries out of necessary values to match the call signature of E_field()
    init_E_field = E_field({"N":init_N, "P":init_P}, 
                           {"rel_permitivity":eps, "N0":n0, "P0":p0, "Node_width":dx})
    #init_E_field = np.zeros(m+1)
    
    init_condition = np.concatenate([init_N, init_P, init_E_field], axis=None)

    if do_ss:
        init_dN = init_N - n0
        init_dP = init_P - p0

    else:
        init_dN = 0
        init_dP = 0

    ## Generate a weight distribution needed for photon recycle term if photon recycle is being considered
    if recycle_photons:
        combined_weight = gen_weight_distribution(m, dx, alpha, delta_frac, 
                                                  back_refl_frac, frac_emitted)
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

    args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, 
            tauN, tauP, B, eps, eps0, q, q_C, kB, 
            recycle_photons, do_ss, alpha, back_refl_frac, 
            delta_frac, frac_emitted, combined_weight, 
            E_field_ext, dEcdz, dChidz, init_dN, 
            init_dP)
    ## Do n time steps
    tSteps = np.linspace(0, n*dt, n+1)
    
    sol = intg.solve_ivp(dydt2, [0,n*dt], init_condition, args=args, t_eval=tSteps, method='BDF', max_step=hmax_)   #  Variable dt explicit
    data = sol.y.T
            
    if write_output:
        ## Prep output files
        with tables.open_file(data_path_name + "-N.h5", mode='a') as ofstream_N, \
            tables.open_file(data_path_name + "-P.h5", mode='a') as ofstream_P:
            array_N = ofstream_N.root.data
            array_P = ofstream_P.root.data
            array_N.append(data[1:,0:m])
            array_P.append(data[1:,m:2*(m)])
            
        return #error_data

    else:
        array_N = data[:,0:m]
        array_P = data[:,m:2*(m)]

        return #array_N, array_P, error_data
    
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
    return (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"]) / ((params["tau_N"] * sim_outputs["P"]) + (params["tau_P"] * sim_outputs["N"]))

def tau_diff(PL, dt):
    """
    Calculates effective particle lifetime from TRPL.

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

def prep_PL(rad_rec, i, j, need_extra_node, params, ignore_recycle):
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
    params : dict {"param name":float or 1D ndarray}
        Collection of parameters from metadata
    ignore_recycle : int (1 or 0)
        Whether to account for photon recycling
    Returns
    -------
    PL_base : 2D ndarray
        PL(x,t)

    """
    
    frac_emitted = params["frac_emitted"]
    alpha = params["alpha"]
    back_refl_frac = params["back_reflectivity"]
    delta = 0 if ignore_recycle else params["delta"]
    dx = params["Node_width"]
    total_length = params["Total_length"]

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
            
        # Sometimes the need_extra_node and to_pos mess up and make the distance
        # array one too long. Patch here and figure it out later.
        if len(distance) > len(rad_rec[0]):
            distance = distance[:len(rad_rec[0])]
    
    PL_base = frac_emitted * (rad_rec * ((1 - delta) + (0.5 * delta * np.exp(-alpha * distance))
                              + (0.5 * delta * back_refl_frac * np.exp(-alpha * (total_length + distance)))))
    
    return PL_base
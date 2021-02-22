#################################################
# Wrapper functions for calculations, as well as
# non-differential equation calculations.
# The differential equations are located in odefuncs.py, 
# while these do the rest of the math.
################################################# 

import tables
import numpy as np
from scipy import integrate as intg
import odefuncs
from utils import to_index, to_pos, to_array

def pulse_laser_power_spotsize(power, spotsize, freq, wavelength, alpha, x_array, hc=6.626e-34*2.997e8):
    # h and c are Planck's const and speed of light, respectively. These default to common units [J*s] and [m/s] but
    # they may be passed in with different units.
    return (power / (spotsize * freq * hc / wavelength) * alpha * np.exp(-alpha * x_array))

def pulse_laser_powerdensity(power_density, freq, wavelength, alpha, x_array, hc=6.626e-34*2.997e8):
    return (power_density / (freq * hc / wavelength) * alpha * np.exp(-alpha * x_array))

def pulse_laser_maxgen(max_gen, alpha, x_array, hc=6.626e-34*2.997e8):
    return (max_gen * np.exp(-alpha * x_array))

def pulse_laser_totalgen(total_gen, total_length, alpha, x_array, hc=6.626e-34*2.997e8):
    return ((total_gen * total_length * alpha * np.exp(alpha * total_length)) / (np.exp(alpha * total_length) - 1) * np.exp(-alpha * x_array))

def gen_weight_distribution(m, dx, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, symmetric=True):
    distance = np.arange(0, m*dx, dx)
    distance_matrix = np.zeros((m, m))
    lf_distance_matrix = np.zeros((m, m)) # Account for "other half" of a symmetric system

    # Each row in distance_matrix represents the weight function centered around a different position
    for i in range(0,m):
        distance_matrix[i] = np.concatenate((np.flip(distance[0:i+1], 0), distance[1:m - i]))
        if symmetric: lf_distance_matrix[i] = distance + ((i+1) * dx)
    
    weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
    if symmetric: weight += np.exp(-(alphaCof + thetaCof) * lf_distance_matrix)
    return alphaCof * 0.5 * (1 - fracEmitted) * delta_frac * weight


def PL_weight_distribution(m, dx, total_length, i, j, alpha, theta, delta, frac_emitted, need_extra_node, symmetric=True):
    distance = np.arange(0, total_length, dx)

    # Make room for the extra node if needed
    if need_extra_node:
        distance_matrix = np.zeros((j - i + 1 + 1, m))
        lf_distance_matrix = np.zeros((j - i + 1 + 1, m))

        # Each row in weight will represent the weight function centered around a different position
        # Additional lf_weight counted in if wire is symmetric
        for n in range(i, j + 1 + 1):
            distance_matrix[n - i] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:m - n]))
            lf_distance_matrix[n - i] = distance + ((n+1) * dx)

    else: # if we don't need the extra node
        distance_matrix = np.zeros((j - i + 1, m))
        lf_distance_matrix = np.zeros((j - i + 1, m))

        for n in range(i, j + 1):
            distance_matrix[n - i] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:m - n]))
            lf_distance_matrix[n - i] = distance + ((n+1) * dx)
    
    weight = np.exp(-(alpha + theta) * distance_matrix)
    weight2 = np.exp(-(theta) * distance_matrix)
    
    if symmetric:
        weight += np.exp(-(alpha + theta) * lf_distance_matrix) 
        weight2 += np.exp(-(theta) * lf_distance_matrix)
        
    return (1 - frac_emitted) * 0.5 * theta * (delta * weight + (1 - delta) * weight2)

def ode_nanowire(data_path_name, m, n, dx, dt, params, recycle_photons=True, symmetric=True, do_ss=False, write_output=True, init_N=0, init_P=0, init_E_field=0):
    ## Problem statement:
    # Create a discretized, time and space dependent solution (N(x,t) and P(x,t)) of the carrier model with m space steps and n time steps
    # Space step size is dx, time step is dt
    # Initial conditions: init_N, init_P, init_E_field
    # Optional photon recycle term

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
    
    ## Set initial condition
    init_condition = np.concatenate([init_N, init_P, init_E_field], axis=None)

    if do_ss:
        init_N_copy = init_N
        init_P_copy = init_P

    else:
        init_N_copy = 0
        init_P_copy = 0

    ## Generate a weight distribution needed for photon recycle term if photon recycle is being considered
    if recycle_photons:
        combined_weight = gen_weight_distribution(m, dx, alphaCof, thetaCof, delta_frac, fracEmitted, symmetric)
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
    # FIXME: hmax as a simulation parameter
    tSteps = np.linspace(0, n*dt, n+1)
    data, error_data = intg.odeint(odefuncs.dydt2, init_condition, tSteps, args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons, do_ss, alphaCof, thetaCof, delta_frac, fracEmitted, combined_weight, E_field_ext, dEcdz, dChidz, init_N_copy, init_P_copy),\
        tfirst=True, full_output=True)
        
    if (data[1:, 0:2*m] < 0).any():
        h = np.geomspace(2**2, 2**-6, 9)
        for hmax in h:
            print("Simulation is not converging well, retrying with hmax={}".format(hmax))
            data, error_data = intg.odeint(odefuncs.dydt2, init_condition, tSteps, args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons, do_ss, alphaCof, thetaCof, delta_frac, fracEmitted, combined_weight, E_field_ext, dEcdz, dChidz, init_N_copy, init_P_copy),\
                                           tfirst=True, full_output=True, hmax=hmax)
                
            if not (data[1:, 0:2*m] < 0).any():
                break
            
    if write_output:
        ## Prep output files
        with tables.open_file(data_path_name + "-N.h5", mode='a') as ofstream_N, \
            tables.open_file(data_path_name + "-P.h5", mode='a') as ofstream_P, \
            tables.open_file(data_path_name + "-E_field.h5", mode='a') as ofstream_E_field:
            array_N = ofstream_N.root.data
            array_P = ofstream_P.root.data
            array_E_field = ofstream_E_field.root.data
            array_N.append(data[1:,0:m])
            array_P.append(data[1:,m:2*(m)])
            array_E_field.append(data[1:,2*(m):])

        return error_data

    else:
        array_N = data[:,0:m]
        array_P = data[:,m:2*(m)]

        return array_N, array_P, error_data

def prep_PL(radRec, i, j, need_extra_node, params):
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
    
    combined_weight = PL_weight_distribution(m, dx, total_length, i, j, alpha, theta, delta, frac_emitted, need_extra_node, params["symmetric_system"])

    for p in range(len(PL_base[0])):
        PL_base[:,p] += intg.trapz(combined_weight[p] * radRec, dx=dx, axis=1).T + radRec[:,i+p] * theta * (1-frac_emitted) * 0.5
    
    return PL_base

def new_integrate(base_data, l_bound, u_bound, dx, total_length, need_extra_node):
    i = to_index(l_bound, dx, total_length)
    j = to_index(u_bound, dx, total_length)
    if base_data.ndim == 1:
        base_data = base_data[None]
    
    if l_bound == u_bound:
        I_base = base_data[:,0]
        if l_bound >= to_pos(i, dx) + dx / 2 and not l_bound == total_length:
            I_plus_one = base_data[:,1]

        if l_bound == to_pos(i, dx) + dx / 2 and not l_bound == total_length:
            I_data = (I_base + I_plus_one) / 2

        elif l_bound > to_pos(i, dx) + dx / 2:
            I_data = I_plus_one

        else:
            I_data = I_base
    else:
        if need_extra_node:
            I_base = base_data
            I_data = intg.trapz(I_base[:, :-1], dx=dx, axis=1)
            
        else:
            I_base = base_data
            I_data = intg.trapz(I_base, dx=dx, axis=1)

        I_data += correct_integral(I_base.T, l_bound, u_bound, i, j, dx)
    return I_data

def correct_integral(integrand, l_bound, u_bound, i, j, dx):
    uncorrected_l_bound = to_pos(i, dx)
    uncorrected_u_bound = to_pos(j, dx)
    lfrac1 = min(l_bound - uncorrected_l_bound, dx / 2)

    # Yes, integrand[0] and not integrand[i]. Note that in integrate(), the ith node maps to integrand[0] and the jth node maps to integrand[j-i].
    l_bound_correction = integrand[0] * lfrac1

    if l_bound > uncorrected_l_bound + dx / 2:
        lfrac2 = (l_bound - (uncorrected_l_bound + dx / 2))
        l_bound_correction += integrand[0+1] * lfrac2

    ufrac1 = min(u_bound - uncorrected_u_bound, dx / 2)
    u_bound_correction = integrand[j-i] * ufrac1

    if u_bound > uncorrected_u_bound + dx / 2:
        ufrac2 = (u_bound - (uncorrected_u_bound + dx / 2))
        u_bound_correction += integrand[j-i+1] * ufrac2

    return u_bound_correction - l_bound_correction

def tau_diff(PL, dt):
    ln_PL = np.log(PL)
    dln_PLdt = np.zeros(ln_PL.__len__())
    dln_PLdt[0] = (ln_PL[1] - ln_PL[0]) / dt
    dln_PLdt[-1] = (ln_PL[-1] - ln_PL[-2]) / dt
    dln_PLdt[1:-1] = (np.roll(ln_PL, -1)[1:-1] - np.roll(ln_PL, 1)[1:-1]) / (2*dt)
    return -(dln_PLdt ** -1)
    
def delta_n(sim_outputs, params):
    return sim_outputs["N"] - params["N0"]

def delta_p(sim_outputs, params):
    return sim_outputs["P"] - params["P0"]

def radiative_recombination(sim_outputs, params):
    return params["B"] * (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"])

def nonradiative_recombination(sim_outputs, params):
    return (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"]) / ((params["Tau_N"] * sim_outputs["P"]) + (params["Tau_P"] * sim_outputs["N"]))

####################

def ode_heatplate(data_path_name, m, n, dx, dt, params, write_output=True):
    atom = tables.Float64Atom()

    ## Unpack params; typecast non-array params to arrays if needed
    q0 = params["Left_flux"]
    qL = params["Right_flux"]
    init_T = to_array(params["init_T"], m, False)

    k = to_array(params["k"], m, False)
    Cp = to_array(params["Cp"], m, False)
    rho = to_array(params["density"], m, False)
    
    init_T[0] = init_T[1] + q0*dx/k[0]
    init_T[-1] = init_T[-2] - qL*dx/k[-1]
    
    tSteps = np.linspace(0, n*dt, n+1)
    data, error_data = intg.odeint(odefuncs.heat_constflux, init_T, tSteps, args=(m, dx, k, rho, Cp, q0, qL),\
        tfirst=True, full_output=True)
            
    if write_output:
        ## Prep output files
        with tables.open_file(data_path_name + "-T.h5", mode='a') as ofstream_T:
            array_T = ofstream_T.root.data

            array_T.append(data[1:])

        return error_data

    else:
        array_T = data

        return array_T, error_data
    return

def heatflux(sim_data, params):
    T = sim_data['T']
    k = to_array(params['k'], len(T), False)
    k_avg = np.zeros(len(T) - 1)
    for i in range(0, len(T) - 1):
        k_avg[i] = (k[i] + k[i+1]) / 2
        
    if T.ndim == 1:
        q = np.zeros(len(T) + 1)
        q[0] = params["Left_flux"]
        q[-1] = params["Right_flux"]
        for i in range(1, len(q) - 1):
            q[i] = -k_avg[i-1] * (T[i] - T[i-1]) / params["Node_width"]
            
    elif T.ndim == 2:
        q = np.zeros((len(T), len(T[0]) + 1))
        q[:,0] += params["Left_flux"]
        q[:,-1] += params["Right_flux"]
        for i in range(1, len(q) - 1):
            q[:,i] = -k_avg[i-1] * (T[:,i] - T[:,i-1]) / params["Node_width"]
        
    return q
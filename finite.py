#################################################
# Wrapper functions for calculations
# The real math is located in odefuncs.py, while
# these do support work like saving math results
# to files.
################################################# 

import tables
import numpy as np
from scipy import integrate as intg
import odefuncs

def check_valid_dx(length, dx):
    return (dx <= length)

def toIndex(x,dx, absUpperBound, is_edge=False):
    absLowerBound = dx / 2 if not is_edge else 0
    if (x < absLowerBound):
        return 0

    if (x > absUpperBound):
        return int(absUpperBound / dx)

    return int((x - absLowerBound) / dx)

def do_AIC(A0, Eg, Exc, Inj, x_array):
    return odefuncs.AIC(A0, Eg, Exc, Inj, x_array)


def toCoord(i,dx, is_edge=False):
    absLowerBound = dx / 2 if not is_edge else 0
    return (absLowerBound + i * dx)

def ode_nanowire(full_path_name,file_name_base, m, n, dx, dt, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, E_field_ext=0, init_N=0, init_P=0, init_E_field=0, init_Ec=0, init_Chi=0):
    ## Problem statement:
    # Create a discretized, time and space dependent solution (N(x,t) and P(x,t)) of the carrier model with m space steps and n time steps
    # Space step size is dx, time step is dt
    # Initial conditions: init_N, init_P, init_E_field
    # Optional photon recycle term

    ## Set data type of array files
    atom = tables.Float64Atom()

    ## Define constants
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
        combined_weight = odefuncs.gen_weight_distribution(m, dx, alphaCof, thetaCof, delta_frac, fracEmitted)
    else:
        combined_weight = 0

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

    print("do_ss was {}".format(do_ss))
    ## Prep output files

    with tables.open_file(full_path_name + "\\" + file_name_base + "-n.h5", mode='a') as ofstream_N, \
        tables.open_file(full_path_name + "\\" + file_name_base + "-p.h5", mode='a') as ofstream_P, \
        tables.open_file(full_path_name + "\\" + file_name_base + "-E_field.h5", mode='a') as ofstream_E_field:
        array_N = ofstream_N.root.N
        array_P = ofstream_P.root.P
        array_E_field = ofstream_E_field.root.E_field

        ## Do n time steps
        tSteps = np.linspace(0, n*dt, n+1)
        data, error_data = intg.odeint(odefuncs.dydt, init_condition, tSteps, args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons, do_ss, alphaCof, thetaCof, delta_frac, fracEmitted, combined_weight, E_field_ext, dEcdz, dChidz, init_N_copy, init_P_copy),\
            tfirst=True, full_output=True, hmax=1.0)
        array_N.append(data[1:,0:m])
        array_P.append(data[1:,m:2*(m)])
        array_E_field.append(data[1:,2*(m):])

    return error_data

def propagatingPL(file_name_base, lower, upper, dx, min, max, B, n0, p0, alphaCof, thetaCof, delta_frac, fracEmitted):
    #note: first dimension is time, second dimension is space
    # Always rounds down when converting from coordinate to index; therefore, some corrections to the integration will be needed
    with tables.open_file("Data\\" + file_name_base + "\\" + file_name_base + "-n.h5", mode='r') as ifstream_N, \
        tables.open_file("Data\\" + file_name_base + "\\" + file_name_base + "-p.h5", mode='r') as ifstream_P:
        radRec = B * ((n0 + np.array(ifstream_N.root.N)) * (p0 + np.array(ifstream_P.root.P)) - n0 * p0)

    i = toIndex(lower, dx, max)
    j = toIndex(upper, dx, max)
    if lower == upper: j += 1
    m = int(max / dx)

    distance = np.linspace(0, max - dx, m)

    # Make room for one more value than necessary - this value at index j+1 will be used to pad the upper correction
    distance_matrix = np.zeros((j - i + 1, m))
    lf_distance_matrix = np.zeros((j - i + 1, m))
    rf_distance_matrix = np.zeros((j - i + 1, m))

    # Each row in weight will represent the weight function centered around a different position
    # Total reflection is assumed to occur at either end of the system: 
    # Left (x=0) reflection is equivalent to a symmetric wire situation while right reflection (x=thickness) is usually negligible
    for n in range(i, j + 1):
        distance_matrix[n - i] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:m - n]))
        lf_distance_matrix[n - i] = distance + (n * dx)
        rf_distance_matrix[n - i] = (max - distance) + (max - n * dx)

    weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
    lf_weight = np.exp(-(alphaCof + thetaCof) * lf_distance_matrix)
    rf_weight = np.exp(-(alphaCof + thetaCof) * rf_distance_matrix) * 0

    combined_weight = (1 - fracEmitted) * 0.5 * thetaCof * delta_frac * (weight + lf_weight + rf_weight)

    weight2 = np.exp(-(thetaCof) * distance_matrix)
    lf_weight2 = np.exp(-(thetaCof) * lf_distance_matrix)

    combined_weight2 = (1 - fracEmitted) * 0.5 * thetaCof * (1 - delta_frac) * (weight2 + lf_weight2)

    if (lower == upper):
        # Special case: Integration over a point should yield the value at that point because it's more useful than zero
        # Get that value using linear interpolation on the two nearest indices

        PL_base = fracEmitted * radRec[:,i] + intg.trapz(combined_weight[0] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i] + \
            intg.trapz(combined_weight2[0] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i]
        PL_plusOne = fracEmitted * radRec[:,i+1] + intg.trapz(combined_weight[1] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i+1] + \
            intg.trapz(combined_weight2[1] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i+1]
        
        PL = PL_base + (lower - toCoord(i, dx)) * (PL_plusOne - PL_base) / dx
        
    else:
        # Take a vertical slice of the radRec array from position index i to j+1
        PL_base = fracEmitted * radRec[:, i:j+1]

        for p in range(0,j - i + 1):
            # To each value of the slice add the attenuation contribution with weight centered around that value's corresponding position
            PL_base[:,p] += intg.trapz(combined_weight[p] * radRec, dx=dx, axis=1).transpose() + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i + p] + \
                intg.trapz(combined_weight2[p] * radRec, dx=dx, axis=1).transpose() + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i + p]

        # This yields the integral (the total observed PL) from i to j, but we need the integral from lower to upper:
        # follow it with some corrections using trapezoidal approx.
        PL = intg.trapz(PL_base, dx=dx, axis=1)

        if (lower <= min):
            lowerCorrection= 0
        else:
            lowerCorrection = 0.5 * (lower - toCoord(i,dx)) * (2 * PL_base[:,0] + ((PL_base[:,1] - PL_base[:,0]) * (lower - toCoord(i,dx)) / dx))

        if (upper >= max):
            upperCorrection = 0
        else:
            upperCorrection = 0.5 * (upper - toCoord(j,dx)) * (2 * PL_base[:,-2] + ((PL_base[:,-1] - PL_base[:,-2]) * (upper - toCoord(j,dx)) / dx))

        PL = PL - lowerCorrection + upperCorrection

    return PL

def integrate(base_data, l_bound, u_bound, dx, max):
    i = toIndex(l_bound, dx, max)
    j = toIndex(u_bound, dx, max)
    m = int(max / dx)

    if (l_bound == u_bound):
        # Special case: Integration over a point should yield the value at that point because it's more useful than zero
        # Get that value using linear interpolation on the two nearest indices

        I_base = base_data[:,i]
        I_plus_one = base_data[:,i+1]
        I_data = I_base + (l_bound - toCoord(i, dx)) * (I_plus_one - I_base) / dx

    else:
        I_base = base_data[:, i:j+1]
        I_data = intg.trapz(I_base, dx=dx, axis=1)
        if (l_bound <= 0):
            lower_correction= 0
        else:
            lower_correction = 0.5 * (l_bound - toCoord(i,dx)) * (2 * I_base[:,0] + ((I_base[:,1] - I_base[:,0]) * (l_bound - toCoord(i,dx)) / dx))

        if (u_bound >= max):
            upper_correction = 0
        else:
            upper_correction = 0.5 * (u_bound - toCoord(j,dx)) * (2 * I_base[:,-2] + ((I_base[:,-1] - I_base[:,-2]) * (u_bound - toCoord(j,dx)) / dx))

        I_data = I_data - lower_correction + upper_correction

    return I_data
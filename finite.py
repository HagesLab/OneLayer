#################################################
# Wrapper functions for calculations, as well as
# non-differential equation calculations
# The differential equations are located in odefuncs.py, while
# these do the rest of the math
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

def ode_nanowire(full_path_name, file_name_base, m, n, dx, dt, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, E_field_ext=0, init_N=0, init_P=0, init_E_field=0, init_Ec=0, init_Chi=0, write_output=True):
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

    if write_output:
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

    else:
        ## Do n time steps without writing to output files
        tSteps = np.linspace(0, n*dt, n+1)
        data, error_data = intg.odeint(odefuncs.dydt, init_condition, tSteps, args=(m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons, do_ss, alphaCof, thetaCof, delta_frac, fracEmitted, combined_weight, E_field_ext, dEcdz, dChidz, init_N_copy, init_P_copy),\
            tfirst=True, full_output=True, hmax=1.0)

        array_N = data[:,0:m]
        array_P = data[:,m:2*(m)]

        return array_N, array_P, error_data

def ode_twolayer(m, f, dm, df, thickness_Layer1, thickness_Layer2, z0, dt, total_time, Cn, Cp, tauN, tauP, tauT, tauS, tauD, tauD_FRET, mu_n, mu_p, mu_S, mu_T, n0, p0, Sf, Sb, B, k_fusion, kstar_tgen, Theta_Tgenb, kB, T, T0, q, q_C, eps, eps0, do_Fret, init_N, init_P):
    # FIXME: NEEDS TESTING WITH bay.py
    init_E = np.zeros(m+1)
    init_T = np.zeros(f)
    init_S = np.zeros(f)
    init_D = np.zeros(f)

    init_m = np.linspace(0+dm/2,thickness_Layer1-dm/2,m)
    init_f = np.linspace(0+df/2,thickness_Layer2-df/2,f)
    Face_Space_m = np.linspace(0,thickness_Layer1,m+1)
    Face_Space_f = np.linspace(0,thickness_Layer2,f+1)

    if do_Fret:
        # A small optimization: the weight arrays for DFret never change once the sim has started,
        # So we calculate once and pass into dydt instead of recalculating many times within dydt
        weight_array1 = np.array([1 / (1 + (thickness_Layer1 - i + init_f) ** 3 /z0) for i in init_m])
        weight_array2 = np.array([1 / (1 + (thickness_Layer1 - init_m + i) ** 3 /z0) for i in init_f])

        integrated_weight_array2 = intg.trapz(weight_array2, dx=dm, axis=1)

    else:
        weight_array1 = 0
        integrated_weight_array2 = 0

    init_condition = np.concatenate([init_N, init_P, init_E,init_T,init_S,init_D], axis=None)

    # Do n time steps
    tSteps = np.linspace(0, total_time, total_time/dt+1)

    data, error = intg.odeint(dydt_twolayer, init_condition, tSteps, args=(m, f, dm, df, Cn, Cp, tauN, tauP, tauT, tauS, tauD, tauD_FRET, mu_n, mu_p, mu_S, mu_T, n0, p0, Sf, Sb, B, k_fusion, kstar_tgen, Theta_Tgenb, kB, T, T0, q, q_C, eps, eps0, weight_array1, integrated_weight_array2, do_Fret), tfirst=True, full_output=True)

    #Organize output data into arrays
    array_deltaN = data[:,0:m]
    array_deltaP = data[:,m:2*(m)]
    array_E = data[:,2*(m):3*(m)+1]
    array_deltaT =data[:,3*(m)+1:3*(m)+1+f]
    array_deltaS = data[:,3*(m)+1+f:3*(m)+1+2*(f)]
    array_deltaD = data[:,3*(m)+1+2*(f):]


    #Calcualte integral values from outputs
    Rad_MAPI_array = B*((array_deltaN+n0)*(array_deltaP+p0)-n0*p0)
    PL_Layer1_array = CalcInt(Rad_MAPI_array,dm)
    PL_Layer2_array = CalcInt(array_deltaD / tauD, df)
    TTA_Rate = CalcInt(k_fusion*(array_deltaT+T0)**2,df)
    surface_v_bulk = 9
    TTA_Rate_Surface = CalcInt(k_fusion*(array_deltaT[:surface_v_bulk]+T0)**2,df)
    TTA_Rate_Bulk = CalcInt(k_fusion*(array_deltaT[surface_v_bulk:]+T0)**2,df)

    if (do_ss):
        integrated_init_N = intg.trapz(init_N, dx=dm, axis=0)
        eta_MAPI= PL_Layer1_array / integrated_init_N*100  #[%]
        eta_UC= PL_Layer2_array / integrated_init_N*100    #[%]
    return

def propagatingPL(file_name_base, l_bound, u_bound, dx, min, max, B, n0, p0, alphaCof, thetaCof, delta_frac, fracEmitted, radrec_fromfile=True, rad_rec=0):
    #note: first dimension of radRec is time, second dimension is space
    if radrec_fromfile:
        with tables.open_file("Data\\" + file_name_base + "\\" + file_name_base + "-n.h5", mode='r') as ifstream_N, \
            tables.open_file("Data\\" + file_name_base + "\\" + file_name_base + "-p.h5", mode='r') as ifstream_P:
            radRec = B * ((n0 + np.array(ifstream_N.root.N)) * (p0 + np.array(ifstream_P.root.P)) - n0 * p0)

    else:
        radRec = rad_rec

    i = toIndex(l_bound, dx, max)
    j = toIndex(u_bound, dx, max)
    #if l_bound == u_bound: j += 1
    m = int(max / dx)

    # i and j here illustrate a problem - we would love to integrate from "l_bound" to "u_bound" exactly, but we only have a discrete list of values with spacing dx.
    # The best we can really do is to map our bounds to the greatest space node less than or equal to the bounds, such that for example,
    # with PL data at x = [10, 30, 50, ...] nm,
    # an integral from x = 15 to x = 65 will map to i = 0 (the node at x = 10) and j = 2 (the node at x = 50).

    # Thus, an integral from x = 15 to x = 65 would actually get you the integral from x = 10 to x = 50, but with a principal assumption of finite difference methods - 
    # that the value of each node extends to a radius of dx / 2 around the node,
    # we can correct the integral by subtracting the integral of the x = 10 node from x = 10 to x = 15 and adding in something similar from x = 50 to x = 65.
    # Because node values are constant within their radius, these correction integrals are just the node values multiplied by a distance

    # We may need an extra node if the u_bound bound extends beyond the radius of the highest node - in the above example, we can calculate the portion from x = 50 to x = 60 using PL[j = 2], 
    # but we would need the node at x = 70 to calculate the portion from x = 60 to x = 65 (using PL[3]).
    need_extra_node = u_bound > toCoord(j, dx) + dx / 2 or l_bound == u_bound

    distance = np.linspace(0, max - dx, m)

    # Make room for the extra node if needed
    if need_extra_node:
        distance_matrix = np.zeros((j - i + 1 + 1, m))
        lf_distance_matrix = np.zeros((j - i + 1 + 1, m))
        rf_distance_matrix = np.zeros((j - i + 1 + 1, m))

        # Each row in weight will represent the weight function centered around a different position
        # Total reflection is assumed to occur at either end of the system: 
        # Left (x=0) reflection is equivalent to a symmetric wire situation while right reflection (x=thickness) is usually negligible
        for n in range(i, j + 1 + 1):
            distance_matrix[n - i] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:m - n]))
            lf_distance_matrix[n - i] = distance + (n * dx)
            rf_distance_matrix[n - i] = (max - distance) + (max - n * dx)


    else: # if we don't need the extra node
        distance_matrix = np.zeros((j - i + 1, m))
        lf_distance_matrix = np.zeros((j - i + 1, m))
        rf_distance_matrix = np.zeros((j - i + 1, m))

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

    if (l_bound == u_bound):
        # Technically an integral with identical bounds is zero, but it is far more useful to return the value at that point instead
        # Because of how nodes work, this should just return the value of the appropriate node, or (and this may be controversial) the average of two adjacent nodes if it lies on an edge

        # Even more controversial is how to handle the cases of integrating from x = 0 to x = 0 and vice versa at the far end of the system -
        # these lie on edges but there are no adjacent nodes to average with!
        # We decide for now that x = 0 yields the value of the leftmost node and x = (far end) yields the value of the rightmost node.

        # Ghost nodes are a possibility but eww
        PL_base = fracEmitted * radRec[:,i] + intg.trapz(combined_weight[0] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i] + \
            intg.trapz(combined_weight2[0] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i]

        if l_bound >= toCoord(i, dx) + dx / 2 and not l_bound == max:
            PL_plusOne = fracEmitted * radRec[:,i+1] + intg.trapz(combined_weight[1] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i+1] + \
                intg.trapz(combined_weight2[1] * radRec, dx=dx, axis=1) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i+1]
        

        if l_bound == toCoord(i, dx) + dx / 2 and not l_bound == max: # if the bound lies on an edge and isn't the far end special case
            PL = (PL_base + PL_plusOne) / 2

        elif l_bound > toCoord(i, dx) + dx / 2: # if the bound exceeds the radius of the node it mapped to
            PL = PL_plusOne

        else:
            PL = PL_base
        
    else:
        if need_extra_node:
            # Take a vertical slice of the radRec array from position index i to j + 1
            # This is a little memory optimization to make narrow integrals faster - we could do calculations on the entire radRec and take out a slice corresponding to our bounds,
            # but by slicing radRec first we avoid doing unnecessary calculations on regions of the system we aren't integrating over

            # The downside of this is all the indices of PL_base end up shifted left by i: PL_base[0] contains data regarding the ith node while PL_base[j-i] contains data regarding the jth node
            PL_base = fracEmitted * radRec[:, i:j+1+1]

            for p in range(0,j - i + 1 + 1):
                # To each value of the slice add the attenuation contribution with weight centered around that value's corresponding position
                # And be careful of the fact that radRec is not shifted left like PL_base is
                PL_base[:,p] += intg.trapz(combined_weight[p] * radRec, dx=dx, axis=1).transpose() + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i + p] + \
                    intg.trapz(combined_weight2[p] * radRec, dx=dx, axis=1).transpose() + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i + p]

            # Be careful not to include our "extra" node in the integral - that node is for corrections only
            PL = intg.trapz(PL_base[:, :-1], dx=dx, axis=1)

        else:
            PL_base = fracEmitted * radRec[:, i:j+1]

            for p in range(0,j - i + 1):
                PL_base[:,p] += intg.trapz(combined_weight[p] * radRec, dx=dx, axis=1).transpose() + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * radRec[:,i + p] + \
                    intg.trapz(combined_weight2[p] * radRec, dx=dx, axis=1).transpose() + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * radRec[:,i + p]


            PL = intg.trapz(PL_base, dx=dx, axis=1)

        PL += correct_integral(PL_base.T, l_bound, u_bound, i, j, dx)

    return PL

def integrate(base_data, l_bound, u_bound, dx, max):
    # See propagatingPL() for additional info
    i = toIndex(l_bound, dx, max)
    j = toIndex(u_bound, dx, max)
    m = int(max / dx)

    need_extra_node = u_bound > toCoord(j, dx) + dx / 2

    if (l_bound == u_bound):
        I_base = base_data[:,i]
        I_plus_one = base_data[:,i+1]

        if l_bound == toCoord(i, dx) + dx / 2 and not l_bound == max:
            I_data = (I_base + I_plus_one) / 2

        elif l_bound > toCoord(i, dx) + dx / 2:
            I_data = I_plus_one

        else:
            I_data = I_base

    else:
        if need_extra_node:
            I_base = base_data[:, i:j+1+1]
            I_data = intg.trapz(I_base[:, :-1], dx=dx, axis=1)

        else:
            I_base = base_data[:, i:j+1]
            I_data = intg.trapz(I_base, dx=dx, axis=1)


        I_data += correct_integral(I_base.T, l_bound, u_bound, i, j, dx)
    return I_data

def correct_integral(integrand, l_bound, u_bound, i, j, dx):
    uncorrected_l_bound = toCoord(i, dx)
    uncorrected_u_bound = toCoord(j, dx)
    lfrac1 = min(l_bound - uncorrected_l_bound, dx / 2)

    # Yes, integrand[0] and not integrand[i]. Note that in propagatingPL() and integrate(), the ith node maps to integrand[0] and the jth node maps to integrand[j-i].
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

def CalcInt(input_array,spacing):
    # FIXME: NEEDS TESTING WITH bay.py
    length=np.shape(input_array)[1]
    output_array=np.zeros(length)
    for i in range(length):
        output_array[i]=intg.trapz(input_array[:,i],dx=spacing,axis=-1)
    return output_array
#################################################
# Most of the "math" that TEDs does.
################################################# 
import numpy as np
from scipy import integrate as intg

def AIC(A0, Eg, Exc, Inj, x_array):
    a = (A0 * ((1240 / Exc) - Eg)**0.5)
    return (a * Inj * np.exp(-a * x_array))

def gen_weight_distribution(m, dx, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0):
    distance = np.linspace(0, m*dx, m)
    distance_matrix = np.zeros((m, m))
    lf_distance_matrix = np.zeros((m, m)) # Account for "other half" of a symmetric system

    # Each row in distance_matrix represents the weight function centered around a different position
    for i in range(0,m):
        distance_matrix[i] = np.concatenate((np.flip(distance[0:i+1], 0), distance[1:m - i]))
        lf_distance_matrix[i] = distance + (i * dx)
    
    combined_weight = alphaCof * 0.5 * (1 - fracEmitted) * delta_frac * (np.exp(-(alphaCof + thetaCof) * distance_matrix) + np.exp(-(alphaCof + thetaCof) * lf_distance_matrix))
    #combined_weight = alphaCof * 0.5 * (1 - fracEmitted) * np.exp(-(alphaCof + thetaCof) * distance_matrix) 
    return combined_weight

def dydt(t, y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_N=0, init_P=0):
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

    delta_N = y[0:m]
    delta_P = y[m:2*(m)]
    E_field = y[2*(m):]
    delta_N_edges = (delta_N[:-1] + np.roll(delta_N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME
    delta_P_edges = (delta_P[:-1] + np.roll(delta_P, -1)[:-1]) / 2
    ## Do boundary conditions of Jn, Jp
    # FIXME: Calculate N, P at boundaries?
    Sft = ((delta_N[0] + n0) * (delta_P[0] + p0) - n0 * p0) / (((delta_N[0] + n0) / Sf) + ((delta_P[0] + p0) / Sf))
    Sbt = ((delta_N[m-1] + n0) * (delta_P[m-1] + p0) - n0 * p0) / (((delta_N[m-1] + n0) / Sb) + ((delta_P[m-1] + p0) / Sb))
    Jn[0] = Sft
    Jn[m] = -Sbt
    Jp[0] = -Sft
    Jp[m] = Sbt

    ## Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension, 
    # Jn(t) ~ N(t) * E_field(t) + (dN/dt)
    # np.roll(y,m) shifts the values of array y by m places, allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx) over entire array y
    Jn[1:-1] = (-mu_n * (delta_N_edges + n0) * (q * (E_field[1:-1] + E_field_ext) + dChidz[1:-1]) + 
                (mu_n*kB*T) * ((np.roll(delta_N,-1)[:-1] - delta_N[:-1]) / (dx)))

    ## Changed sign
    Jp[1:-1] = (-mu_p * (delta_P_edges + p0) * (q * (E_field[1:-1] + E_field_ext) + dChidz[1:-1] + dEcdz[1:-1]) -
                (mu_p*kB*T) * ((np.roll(delta_P, -1)[:-1] - delta_P[:-1]) / (dx)))

        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (eps * eps0))

    ## Calculate recombination (consumption) terms
    rad_rec = B * ((delta_N + n0) * (delta_P + p0) - n0 * p0)
    non_rad_rec = ((delta_N + n0) * (delta_P + p0) - n0 * p0) / ((tauN * (delta_P + p0)) + (tauP * (delta_N + n0)))
        
    ## Calculate generation term from photon recycling, if photon recycling is being considered
    if recycle_photons:
        G_array = intg.trapz(rad_rec * combined_weight, dx=dx, axis=1) + (1 - fracEmitted) * 0.5 * alphaCof * delta_frac * rad_rec
    else:
        G_array = 0
    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dx)


    ## delta_N(t) = delta_N(t-1) + dt * (dN/dt)
    #N_new = np.maximum(N_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: dNdt += init_N

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dx)

    ## delta_P(t) = delta_P(t-1) + dt * (dP/dt)
    #P_new = np.maximum(P_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: dPdt += init_P

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt
#################################################
# scipy.odeint() needs a function dydt() to
# call when time stepping forward. Those are
# located here.
################################################# 
import numpy as np
from scipy import integrate as intg
from numba import njit

def dydt2(t, y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_N=0, init_P=0):
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
    Sft = (N[0] * P[0] - n0[0] * p0[0]) / ((N[0] / Sf) + (P[0] / Sf))
    Sbt = (N[m-1] * P[m-1] - n0[m-1] * p0[m-1]) / ((N[m-1] / Sb) + (P[m-1] / Sb))
    Jn[0] = Sft
    Jn[m] = -Sbt
    Jp[0] = -Sft
    Jp[m] = Sbt

    ## Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension, 
    # Jn(t) ~ N(t) * E_field(t) + (dN/dt)
    # np.roll(y,m) shifts the values of array y by m places, allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx) over entire array y
    Jn[1:-1] = (-mu_n[1:-1] * (N_edges) * (q * (E_field[1:-1] + E_field_ext[1:-1]) + dChidz[1:-1]) + 
                (mu_n[1:-1]*kB*T[1:-1]) * ((np.roll(N,-1)[:-1] - N[:-1]) / (dx)))

    ## Changed sign
    Jp[1:-1] = (-mu_p[1:-1] * (P_edges) * (q * (E_field[1:-1] + E_field_ext[1:-1]) + dChidz[1:-1] + dEcdz[1:-1]) -
                (mu_p[1:-1]*kB*T[1:-1]) * ((np.roll(P, -1)[:-1] - P[:-1]) / (dx)))

        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (eps * eps0))
    
    ## Calculate recombination (consumption) terms
    rad_rec = B * (N * P - n0 * p0)
    non_rad_rec = (N * P - n0 * p0) / ((tauN * P) + (tauP * N))
        
    ## Calculate generation term from photon recycling, if photon recycling is being considered
    if recycle_photons:
        G_array = intg.trapz(rad_rec * combined_weight, dx=dx, axis=1) + (1 - fracEmitted) * 0.5 * alphaCof * delta_frac * rad_rec
    else:
        G_array = 0
    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dx)

    ## N(t) = N(t-1) + dt * (dN/dt)
    #N_new = np.maximum(N_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: dNdt += init_N

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dx)

    ## P(t) = P(t-1) + dt * (dP/dt)
    #P_new = np.maximum(P_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec + G_array)
    if do_ss: dPdt += init_P

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt

@njit
def dydt(t, y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_N=0, init_P=0):

    Jn = np.zeros((m+1))
    Jp = np.zeros((m+1))
    dydt = np.zeros(3*m+1)

    N = y[0:m]
    P = y[m:2*(m)]
    E_field = y[2*(m):]
    NP = N * P - n0 * p0

    Sft = Sf * (NP[0]) / (N[0] + P[0])
    Sbt = Sb * (NP[m-1]) / (N[m-1] + P[m-1])
    Jn[0] = -Sft
    Jn[m] = Sbt
    Jp[0] = Sft
    Jp[m] = -Sbt
        
    for i in range(1, len(Jn) - 1):
        Jn[i] = (-mu_n[i] * 0.5 * (N[i-1] + N[i]) * (q * (E_field[i] + E_field_ext[i]) + dChidz[i]) - 
                  (mu_n[i]*kB*T[i]) * (N[i] - N[i-1]) / dx)
        
        Jp[i] = (-mu_p[i] * 0.5 * (P[i-1] + P[i]) * (q * (E_field[i] + E_field_ext[i]) + dChidz[i] + dEcdz[i]) + 
                  (mu_p[i]*kB*T[i]) * (P[i] - P[i-1]) / dx)
        
    # [V nm^-1 ns^-1]
    for i in range(len(Jn)):
        dydt[2*m+i] = (Jn[i] + Jp[i]) * ((q_C) / (eps[i] * eps0))
    
    ## RR
    sink = B * NP

    ## Calculate generation term from photon recycling, if photon recycling is being considered
    if recycle_photons:
        G_array = 0.5 * sink[0] * combined_weight[:,0]
        for i in range(1, len(sink) - 1):
            G_array += sink[i] * combined_weight[:,i]
        G_array += 0.5 * sink[-1] * combined_weight[:,-1]
        G_array *= dx

        G_array += (1 - fracEmitted) * 0.5 * alphaCof * delta_frac * sink

        sink -= G_array
        
    # NRR
    sink += NP / ((tauN * P) + (tauP * N))

    for i in range(len(Jn) - 1):
        dydt[i] = (-Jn[i+1] + Jn[i]) / dx - sink[i]
        dydt[m+i] = (Jp[i+1] - Jp[i]) / dx - sink[i]
        
    if do_ss:
        dydt[:m] += init_N
        dydt[m:2*m] += init_P
        
    return dydt

def dydt_twolayer(t, y, m, f, dm, df, Cn, Cp, tauN, tauP, tauT, tauS, tauD, tauD_FRET, mu_n, mu_p, mu_S, mu_T, n0, p0, Sf, Sb, B, k_fusion, kstar_tgen, Theta_Tgenb, kB, T, T0, q, q_C, eps, eps0, weight_array1, integrated_weight_array2, do_Fret, init_N, init_P):
    #Variables in Layer 1 (MAPI)
    #m nodes, m+1 faces
    #nodes
    delta_N = y[0:m]
    delta_P = y[m:2*(m)]

    #faces
    E_field = y[2*(m):3*(m)+1]
    delta_N_edges = (delta_N[:-1] + np.roll(delta_N, -1)[:-1]) / 2 # Excluding the boundaries
    delta_P_edges = (delta_P[:-1] + np.roll(delta_P, -1)[:-1]) / 2
    Jn = np.zeros((m+1))
    Jp = np.zeros((m+1))

    #Variables in Layer 2 (Rubrene:DBP)
    #f nodes, f+1 faces
    #nodes
    delta_T = y[3*(m)+1:3*(m)+1+f]
    delta_S = y[3*(m)+1+f:3*(m)+1+2*(f)]
    delta_D = y[3*(m)+1+2*(f):]

    #faces
    JT = np.zeros((f+1))
    JS = np.zeros((f+1))

    #Boundary conditions for J
    #Layer 1
    Sft = ((delta_N[0] + n0) * (delta_P[0] + p0) - n0 * p0) / (((delta_N[0] + n0) / Sf) + ((delta_P[0] + p0) / Sf))
    Sbt = ((delta_N[m-1] + n0) * (delta_P[m-1] + p0) - n0 * p0) / (((delta_N[m-1] + n0) / Sb) + ((delta_P[m-1] + p0) / Sb))
    Tgen_surface = kstar_tgen*((delta_N[m-1] + n0) * (delta_P[m-1] + p0) - n0 * p0)
    Jn[0] = Sft
    Jn[m] = -Sbt #- Tgen_surface
    Jp[0] = -Sft
    Jp[m] = Sbt #+ Tgen_surface

    #Layer 2
    JT[0] =   -Theta_Tgenb*Sbt
    #JT[0] =   -Tgen_surface
    JT[f] = 0.
    JS[0] = 0.
    JS[f] = 0.

    # Calculate J, E, & dJ
    # Layer 1
    Jn[1:-1] = (-mu_n * (delta_N_edges + n0) * (E_field[1:-1] ) + (mu_n*kB*T) * ((np.roll(delta_N,-1)[:-1] - delta_N[:-1] ) /(dm)))
    Jp[1:-1] = (-mu_p * (delta_P_edges + p0) * (E_field[1:-1]) - (mu_p*kB*T) * (( np.roll(delta_P, -1)[:-1] - delta_P[:-1] ) / (dm)))

    dJn = (np.roll(Jn, -1)[:-1] -  Jn[:-1]) / (dm)
    dJp = (np.roll(Jp, -1)[:-1] - Jp[:-1] ) / (dm)

    # Layer 2
    JT[1:-1] = (mu_T*kB*(T+T0)) * ((np.roll(delta_T,-1)[:-1] - delta_T[:-1]) /(df))
    JS[1:-1] = (mu_S*kB*(T+T0)) * ((np.roll(delta_S,-1)[:-1] - delta_S[:-1]) /(df))
    dJT = (np.roll(JT, -1)[:-1] - JT[:-1] ) / (df)
    dJS = (np.roll(JS, -1)[:-1] -  JS[:-1] ) / (df)

    # Calculate recombination terms
    #Layer 1
    aug_rec_n = Cn*(delta_N + n0)*((delta_N + n0) * (delta_P + p0) - n0 * p0)
    aug_rec_p = Cp*(delta_P + n0)*((delta_N + n0) * (delta_P + p0) - n0 * p0)
    rad_rec = B * ((delta_N + n0) * (delta_P + p0) - n0 * p0)
    non_rad_rec = ((delta_N + n0) * (delta_P + p0) - n0 * p0) / ((tauN * (delta_P + p0)) + (tauP * (delta_N + n0)))

    #Layer 2
    T_Rec = (delta_T+T0)/tauT
    T_Fusion = k_fusion*(delta_T+T0)**2
    S_Fret = delta_S/tauS
    D_Rec = delta_D/tauD

    # Calculate D Fretting: D_Rec times a distance-dependent weight function
    if (do_Fret):
        # The global keyword allows a function to edit a variable defined outside of its scope

        # Layer 1
        # The nth row of this 2D weight array belongs to the nth node of Layer 1.
        # The mth value in that row belongs to the mth node of Layer 2.
        # The first value of the last row of weight_array, for example, is the proportion of D_Rec
        # transmitted from the leftmost node of Layer 2 to the rightmost node of Layer 1.
        # weight_array1 = [1 / (1 + (thickness_Layer1 - i + init_f) ** 3 /z0) for i in init_m]

        # "Row-wise" multiplication of D_Rec (a 1D array of length = init_f's length) onto each row of 2D weight_array,
        # followed by squeezing the product into a single vertical stack of integration results for Layer 1 nodes
        # (Not really vertical because D_Fret1 is a 1D array without up or down but that mental image helps explain why axis=1 here)
        # See https://urldefense.proofpoint.com/v2/url?u=https-3A__docs.scipy.org_doc_numpy_reference_generated_numpy.trapz.html&d=DwIGaQ&c=sJ6xIWYx-zLMB3EPkvcnVg&r=srGA7w6hFtH6Bov0hksDApkUha590QWTEyQnPqPfL-8&m=mDtcRttxlquRZRlmnBRz75t4OmiADXKLNpt7tgDpi_8&s=_ZSw_cv0x98SMIbnbLq3nNH5_FFwhR6rpbbN63LwBGU&e=  for a quick example
        D_Fret1 = intg.trapz(weight_array1 * delta_D / tauD_FRET, dx=df, axis=1)
        
        # Layer 2
        # Now the rows belong to Layer 2 nodes and we're integrating across Layer 1.
        # It doesn't really matter whether rows are assigned to Layer 1 or 2 as long as the axis= option is correct,
        # but I like giving the variable of integration the inner dimension
        # weight_array2 = [1 / (1 + (thickness_Layer1 - init_m + i) ** 3 /z0) for i in init_f]
        #D_Fret2 = intg.trapz(weight_array2 * delta_D / tauD_FRET, dx=dm, axis=1)
        D_Fret2 = (delta_D / tauD_FRET) * integrated_weight_array2

    else:
        D_Fret1 = 0
        D_Fret2 = 0
    # Differntials
    #Layer 1

    if (do_ss):
        dNdt = ((1/q) * dJn - rad_rec - non_rad_rec - aug_rec_n - aug_rec_p + D_Fret1 + init_N)
        dPdt = ((1/q) * -dJp - rad_rec - non_rad_rec - aug_rec_n - aug_rec_p + D_Fret1 + init_P)
    else:
        dNdt = ((1/q) * dJn - rad_rec - non_rad_rec - aug_rec_n - aug_rec_p + D_Fret1)
        dPdt = ((1/q) * -dJp - rad_rec - non_rad_rec - aug_rec_n - aug_rec_p + D_Fret1)

    dEdt = (Jn + Jp) * ((q_C) / (eps * eps0))

    #Layer 2
    dTdt = ((1/q) * dJT - T_Fusion - T_Rec)
    dSdt = ((1/q) * dJS + T_Fusion - S_Fret)
    dDdt = S_Fret - D_Fret2 - D_Rec
# Package
    dydt = np.concatenate([dNdt, dPdt, dEdt,dTdt,dSdt,dDdt], axis=None)
    return dydt

#@njit(cache=True)
def heat_constflux(t, y, m, dx, k, rho, Cp, q0, qL):
    alpha = k * rho / Cp
    G = alpha / (dx**2)
    T = y
    
    dydt = np.zeros(m)
    for i in range(1, len(T) - 1):
        dydt[i] = G[i+1]*T[i+1] - 2*G[i]*T[i] + G[i-1]*T[i-1]
    
    # Bounds
    dydt[0] = dydt[1]
    dydt[-1] = dydt[-2]
    
    return dydt
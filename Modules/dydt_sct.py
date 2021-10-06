# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
import numpy as np
from scipy import integrate as intg

q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]


def dydt_sct(t, y, m, f, dm, df, Cn, Cp, 
         tauN, tauP, tauT, tauS, tauD, 
         mu_n, mu_p, mu_s, mu_T,
         n0, p0, T0, Sf, Sb, St, B, k_fusion, k_0, mapi_temperature, rubrene_temperature,
         eps, 
         mu_n_up, mu_q, Ssct, Sn, Sp, W_CB, W_VB, 
         weight1=0, weight2=0, do_Fret=False, do_ss=False, 
         init_dN=0, init_dP=0):

    # W_VB = 0.1                  #[eV]
    # Ssct = 100*(1e14)           #St=100 #nm/s
    # Sp = 5 * 1e3 * 1e7          # cm to nm,  STn = 5.3·10 cm s-1
    # mu_q = 2 * (1e14)           # cm2/Vs to nm2/Vs

    # mu_q = mu_T * 0 + mu_q      # transforming into array, for some reason mu are arrays
    
    """Derivative function for two-layer carrier model."""
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((m+1))
    Jp = np.zeros((m+1))
    JT = np.zeros((f+1))
    JS = np.zeros((f+1))
    Jq = np.zeros((f+1))
    

    # Unpack simulated variables
    # y = 0 - N - m - P - 2m - E_field - 3m - delta_T - 3m+f - delta_S - 3m+2f - delta_D - 3m+3f - Q - 3m+4f
    N = y[0:m]
    P = y[m:2*(m)]
    E_field = y[2*(m):3*(m)+1]
    delta_T = y[3*(m)+1:3*(m)+1+f]
    delta_S = y[3*(m)+1+f:3*(m)+1+2*(f)]
    delta_D = y[3*(m)+1+2*(f):3*(m)+1+3*(f)]
    Q = y[3*(m)+1+3*(f):]

    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME

    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    
    
    # MAPI boundaries
    Sft = Sf * (N[0] * P[0] - n0[0] * p0[0]) / (N[0] + P[0])                # surf. recomb. front
    
    
    Sbt = Sb * (N[m-1] * P[m-1] - n0[m-1] * p0[m-1]) / (N[m-1] + P[m-1])    # surf. recomb. back
    
    Stt = Ssct * (N[m-1] * Q[0])                                            # seq. charge transfer triplet generation (SCTG)
    #Stt = St * (N[m-1] * P[m-1] - n0[m-1] * p0[m-1]) / (N[m-1] + P[m-1])    # surf. recomb. triplet generation (SRTG)
    Spt = Sp * (P[m-1] - Q[0] * np.exp(-W_VB / (kB*rubrene_temperature[0])))    # charge transfer of holes into rubrene

    Jn[0] = Sft                                                             # electron current front
    Jn[m] = -(Sbt+Stt)                                                      # electron current back (interface)
    Jp[0] = -Sft                                                            # hole current front
    # Jp[m] = (Sbt+Stt)                                                       # hole current back (interface)
    Jp[m] = (Sbt+Stt+Spt)                                                       # hole current back (interface)
    
    # Rubrene boundaries
    JT[0] = -Stt                                                            # triplet "current" (flux) at interface
    JT[f] = 0
    JS[0] = 0
    JS[f] = 0
    Jq[0] = -Spt+Stt

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
    ## Rubrene Jq flux (q = holes in rubrene vs p = holes in MAPI)
    ## Not account E_field
    Jq[1:-1] = (mu_q[1:-1]*kB*rubrene_temperature[1:-1]) * ((np.roll(Q,-1)[:-1] - Q[:-1]) / (df))

    dJT = (np.roll(JT, -1)[:-1] - JT[:-1]) / (df)
    dJS = (np.roll(JS, -1)[:-1] - JS[:-1]) / (df)
    ## Rubrene dJq
    dJq = (np.roll(Jq, -1)[:-1] - Jq[:-1]) / (df)


    
    ## Calculate recombination (consumption) terms
    # MAPI Auger + RR + SRH
    n_rec = (Cn*N + Cp*P + B + (1 / ((tauN * P) + (tauP * N)))) * (N * P - n0 * p0)
    p_rec = n_rec
        
    # Rubrene single- and bi-molecular decays
    T_rec = delta_T / tauT
    T_fusion = k_fusion * (delta_T) ** 2
    S_Fret = delta_S / tauS
    D_rec = delta_D / tauD                                                 # D_rec? delta_D ?
    
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
    dQdt = ((1/q) * dJq) # - T_TCA


    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt, dTdt, dSdt, dDdt, dQdt], axis=None)
    return dydt
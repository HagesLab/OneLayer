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


def dydt_basic(t, y, g, p, wt_fret_to_mapi=0, wt_fret_from_rubrene=0, do_Fret=False, do_ss=False, 
         init_dN=0, init_dP=0):
    """Derivative function for two-layer carrier model."""
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((g.mapi_nx+1))
    Jp = np.zeros((g.mapi_nx+1))
    JT = np.zeros((g.rubrene_nx+1))
    JS = np.zeros((g.rubrene_nx+1))

    # Unpack simulated variables
    N = y[0:g.mapi_nx]
    P = y[g.mapi_nx:2*(g.mapi_nx)]
    E_field = y[2*(g.mapi_nx):3*(g.mapi_nx)+1]
    delta_T = y[3*(g.mapi_nx)+1:3*(g.mapi_nx)+1+g.rubrene_nx]
    delta_S = y[3*(g.mapi_nx)+1+g.rubrene_nx:3*(g.mapi_nx)+1+2*(g.rubrene_nx)]
    delta_D = y[3*(g.mapi_nx)+1+2*(g.rubrene_nx):]
    
    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    
    # MAPI boundaries
    Sft = p.Sf * (N[0] * P[0] - p.n0[0] * p.p0[0]) / (N[0] + P[0])
    Sbt = p.Sb * (N[g.mapi_nx-1] * P[g.mapi_nx-1] - p.n0[g.mapi_nx-1] * p.p0[g.mapi_nx-1]) / (N[g.mapi_nx-1] + P[g.mapi_nx-1])
    Stt = p.St * (N[g.mapi_nx-1] * P[g.mapi_nx-1] - p.n0[g.mapi_nx-1] * p.p0[g.mapi_nx-1]) / (N[g.mapi_nx-1] + P[g.mapi_nx-1])
    Jn[0] = Sft
    Jn[g.mapi_nx] = -(Sbt+Stt)
    Jp[0] = -Sft
    Jp[g.mapi_nx] = (Sbt+Stt)
    
    # Rubrene boundaries
    JT[0] = -Stt
    JT[g.rubrene_nx] = 0
    JS[0] = 0
    JS[g.rubrene_nx] = 0

    ## Calculate Jn, Jp [nm^-2 ns^-1] for MAPI, 
    Jn[1:-1] = (-p.mu_n[1:-1] * (N_edges) * (q * E_field[1:-1]) 
                + (p.mu_n[1:-1]*kB*p.mapi_temperature[1:-1]) * ((np.roll(N,-1)[:-1] - N[:-1]) / (g.mapi_dx)))

    Jp[1:-1] = (-p.mu_p[1:-1] * (P_edges) * (q * E_field[1:-1]) 
                - (p.mu_p[1:-1]*kB*p.mapi_temperature[1:-1]) * ((np.roll(P, -1)[:-1] - P[:-1]) / (g.mapi_dx)))

    dJn = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (g.mapi_dx)
    dJp = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (g.mapi_dx)
        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (p.eps * eps0))
    
    ## Rubrene J fluxes
    JT[1:-1] = (p.mu_T[1:-1]*kB*p.rubrene_temperature[1:-1]) * ((np.roll(delta_T,-1)[:-1] - delta_T[:-1]) / (g.rubrene_dx))
    JS[1:-1] = (p.mu_s[1:-1]*kB*p.rubrene_temperature[1:-1]) * ((np.roll(delta_S,-1)[:-1] - delta_S[:-1]) / (g.rubrene_dx))
    dJT = (np.roll(JT, -1)[:-1] - JT[:-1]) / (g.rubrene_dx)
    dJS = (np.roll(JS, -1)[:-1] - JS[:-1]) / (g.rubrene_dx)
    
    ## Calculate recombination (consumption) terms
    # MAPI Auger + RR + SRH
    n_rec = (p.Cn*N + p.Cp*P + p.B + (1 / ((p.tauN * P) + (p.tauP * N)))) * (N * P - p.n0 * p.p0)
    p_rec = n_rec
        
    # Rubrene single- and bi-molecular decays
    T_rec = delta_T / p.tauT
    T_fusion = p.k_fusion * (delta_T) ** 2
    S_Fret = delta_S / p.tauS
    D_rec = delta_D / p.tauD
    
    ## Calculate D_Fretting
    if do_Fret:
        D_Fret1 = intg.trapz(wt_fret_to_mapi * delta_D * p.k_0 / p.tauD, dx=g.rubrene_dx, axis=1)
        D_Fret2 = (delta_D * p.k_0 / p.tauD) * wt_fret_from_rubrene
        
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


def dydt_sct(t, y, g, p,
         wt_fret_to_mapi=0, wt_fret_from_rubrene=0, do_Fret=False, do_ss=False, 
         init_dN=0, init_dP=0):

    """Derivative function for two-layer carrier model."""
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((g.mapi_nx+1))
    Jp = np.zeros((g.mapi_nx+1))
    JT = np.zeros((g.rubrene_nx+1))
    JS = np.zeros((g.rubrene_nx+1))
    Jq = np.zeros((g.rubrene_nx+1))
    

    # Unpack simulated variables
    # y = 0 - N - m - P - 2m - E_field - 3m - delta_T - 3m+f - delta_S - 3m+2f - delta_D - 3m+3f - Q - 3m+4f
    N = y[0:g.mapi_nx]
    P = y[g.mapi_nx:2*(g.mapi_nx)]
    E_field = y[2*(g.mapi_nx):3*(g.mapi_nx)+1]
    delta_T = y[3*(g.mapi_nx)+1:3*(g.mapi_nx)+1+g.rubrene_nx]
    delta_S = y[3*(g.mapi_nx)+1+g.rubrene_nx:3*(g.mapi_nx)+1+2*(g.rubrene_nx)]
    delta_D = y[3*(g.mapi_nx)+1+2*(g.rubrene_nx):3*(g.mapi_nx)+1+3*(g.rubrene_nx)]
    Q = y[3*(g.mapi_nx)+1+3*(g.rubrene_nx):3*g.mapi_nx+1 + 4*g.rubrene_nx]
    E_upc = y[3*g.mapi_nx+1 + 4*g.rubrene_nx:]

    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME

    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    T_edges = (delta_T[:-1] + np.roll(delta_T, -1)[:-1]) / 2
    S_edges = (delta_S[:-1] + np.roll(delta_S, -1)[:-1]) / 2
    Q_edges = (Q[:-1] + np.roll(Q, -1)[:-1]) / 2
    
    # MAPI boundaries
    Sft = p.Sf * (N[0] * P[0] - p.n0[0] * p.p0[0]) / (N[0] + P[0])                # surf. recomb. front
    
    Sbt = p.Sb * (N[g.mapi_nx-1] * P[g.mapi_nx-1] - p.n0[g.mapi_nx-1] * p.p0[g.mapi_nx-1]) / (N[g.mapi_nx-1] + P[g.mapi_nx-1])    # surf. recomb. back
    
    Stt = p.Ssct * (N[g.mapi_nx-1] * Q[0])                                            # seq. charge transfer triplet generation (SCTG)
    
    Spt = p.Sp * (P[g.mapi_nx-1] - Q[0] * np.exp(-p.w_vb / (kB*p.rubrene_temperature[0])))    # charge transfer of holes into rubrene

    Jn[0] = Sft                                                             # electron current front
    Jn[g.mapi_nx] = -(Sbt+Stt)                                                      # electron current back (interface)
    Jp[0] = -Sft                                                            # hole current front
    #Jp[g.mapi_nx] = (Sbt+Spt)                                                       # hole current back (interface)
    Jp[g.mapi_nx] = (Sbt+Stt+Spt)                                                       # hole current back (interface)
    
    # Rubrene boundaries
    JT[0] = -Stt                                                            # triplet "current" (flux) at interface
    JT[g.rubrene_nx] = 0
    JS[0] = 0
    JS[g.rubrene_nx] = 0
    #Jq[0] = -Spt+Stt
    Jq[0] = Spt-Stt

    Jn[1:-1] = (-p.mu_n[1:-1] * (N_edges) * (q * E_field[1:-1]) 
                + (p.mu_n[1:-1]*kB*p.mapi_temperature[1:-1]) * ((np.roll(N,-1)[:-1] - N[:-1]) / (g.mapi_dx)))
    Jp[1:-1] = (-p.mu_p[1:-1] * (P_edges) * (q * E_field[1:-1]) 
                - (p.mu_p[1:-1]*kB*p.mapi_temperature[1:-1]) * ((np.roll(P, -1)[:-1] - P[:-1]) / (g.mapi_dx)))
    dJn = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (g.mapi_dx)
    dJp = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (g.mapi_dx)
        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (p.eps * eps0))
    
    ## Rubrene J fluxes
    JT[1:-1] = (-p.mu_T[1:-1] * (T_edges) * (q * E_upc[1:-1]) 
               + (p.mu_T[1:-1]*kB*p.rubrene_temperature[1:-1]) * ((np.roll(delta_T,-1)[:-1] - delta_T[:-1]) / (g.rubrene_dx)))
    JS[1:-1] = (-p.mu_s[1:-1] * (S_edges) * (q * E_upc[1:-1]) 
               + (p.mu_s[1:-1]*kB*p.rubrene_temperature[1:-1]) * ((np.roll(delta_S,-1)[:-1] - delta_S[:-1]) / (g.rubrene_dx)))
    ## Rubrene Jq flux (q = holes in rubrene vs p = holes in MAPI)
    ## Not account E_field
    Jq[1:-1] = (-p.mu_p_up[1:-1] * (Q_edges) * (q * E_upc[1:-1]) 
                -(p.mu_p_up[1:-1]*kB*p.rubrene_temperature[1:-1]) * ((np.roll(Q,-1)[:-1] - Q[:-1]) / (g.rubrene_dx)))

    dJT = (np.roll(JT, -1)[:-1] - JT[:-1]) / (g.rubrene_dx)
    dJS = (np.roll(JS, -1)[:-1] - JS[:-1]) / (g.rubrene_dx)
    ## Rubrene dJq
    dJq = (np.roll(Jq, -1)[:-1] - Jq[:-1]) / (g.rubrene_dx)


    #dE_ucdt = np.zeros_like(E_upc)
    dE_ucdt = (JT + JS + Jq) * ((q_C) / (p.uc_eps * eps0))

    
    ## Calculate recombination (consumption) terms
    # MAPI Auger + RR + SRH
    n_rec = (p.Cn*N + p.Cp*P + p.B + (1 / ((p.tauN * P) + (p.tauP * N)))) * (N * P - p.n0 * p.p0)
    p_rec = n_rec
        
    # Rubrene single- and bi-molecular decays
    T_rec = delta_T / p.tauT
    T_fusion = p.k_fusion * (delta_T) ** 2
    S_Fret = delta_S / p.tauS
    D_rec = delta_D / p.tauD                                                 # D_rec? delta_D ?
    
    ## Calculate D_Fretting
    if do_Fret:
        D_Fret1 = intg.trapz(wt_fret_to_mapi * delta_D * p.k_0 / p.tauD, dx=g.rubrene_dx, axis=1)
        D_Fret2 = (delta_D * p.k_0 / p.tauD) * wt_fret_from_rubrene
        
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
    dQdt = ((1/q) * -dJq) # - T_TCA


    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt, dTdt, dSdt, dDdt, dQdt, dE_ucdt], axis=None)
    return dydt

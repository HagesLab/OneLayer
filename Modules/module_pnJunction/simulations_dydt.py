# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
import numpy as np

q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]
from Modules.module_pnJunction.calculations import V_poisson


def dydt_multi_volts(t,y,G,PA, s, do_ss):
    #Need to tell the function which values of y correspond to n, p, and E
    n  = y[0:G.n_total]
    p = y[G.n_total:2*G.n_total]
    
    #Define n,p at interior faces
    n_faces = ((n + np.roll(n,-1))/2)[:-1]   #Finds average of i and (i+1) - delete last point becuase (i+1) becomes index 0
    p_faces = ((p + np.roll(p,-1))/2)[:-1]
    
    
    #Flux (current) terms at all faces
    Jn = np.zeros(G.n_total+1)
    Jp = np.zeros(G.n_total+1)
    
    V = V_poisson(G.dx, n, p, PA.n0, PA.p0, PA.eps, PA.V0, PA.VL)
    
    dVdx = (np.roll(V,-1) - V)[:-1] / G.inter_dx

    
    dEfn = ( kB*PA.temperature[1:-1]*(np.roll(n,-1)-n)[:-1] / G.inter_dx) - q*n_faces*(dVdx+PA.dchidx)    #[eV m*(np.roll(V,-1) - V)[:-1] / G.inter_dx*-1]
    dEfp = (-kB*PA.temperature[1:-1]*(np.roll(p,-1)-p)[:-1] / G.inter_dx) - q*p_faces*(dVdx+PA.dchidx+PA.dEgdx)   #[eV m**-1]
    
    # J at boundaries
    Jn[0], Jn[G.n_total], Jp[0], Jp[G.n_total] = 0, 0, 0, 0   #[(eV/V) m**-2 s**-1]


    #Define J at interior faces
    Jn[1:-1] = PA.mu_n[1:-1]*dEfn     #[(eV/V) m**-2 s**-1]
    Jp[1:-1] = PA.mu_p[1:-1]*dEfp     #[(eV/V) m**-2 s**-1]
    if t != PA.t_old:
        PA.t_old = t
        print(t)

    # Calculate flux difference
    dJn = ((np.roll(Jn,-1) - Jn)[:-1] /G.dx)  #[(eV/V) m**-3 s**-1]
    dJp = ((np.roll(Jp,-1) - Jp)[:-1] /G.dx)  #[(eV/V) m**-3 s**-1]

    rr_srh = (n * p - PA.n0 * PA.p0) / (PA.tauN * p + PA.tauP * n)
    rr_rad = PA.B * (n * p - PA.n0 * PA.p0)
    rr_aug = (PA.Cn * n + PA.Cp * p) * (n * p - PA.n0 * PA.p0)
    
    rr_rs = np.zeros_like(rr_srh)
    
    # First "interface" (front surface)
    rr_rs[0] = (PA.RS[0] / G.dx[0]) * (n[0] * p[0] - PA.n0[0] * PA.p0[0]) / (n[0] + p[0]) 
    
    # Last "interface" (back surface)
    rr_rs[-1] = (PA.RS[-1] / G.dx[-1]) * (n[-1] * p[-1] - PA.n0[-1] * PA.p0[-1]) / (n[-1] + p[-1])
    
    # Intermediate interfaces
    for i, rbound in enumerate(G.nx_bounds[1:-1], 1):
        lbound = rbound - 1
        rr_rs[lbound] = (PA.RS[i][0] / G.dx[lbound]) * (n[lbound] * p[lbound] - PA.n0[lbound] * PA.p0[lbound]) / (n[lbound] + p[lbound])
        rr_rs[rbound] = (PA.RS[i][1] / G.dx[rbound]) * (n[rbound] * p[rbound] - PA.n0[rbound] * PA.p0[rbound]) / (n[rbound] + p[rbound])

    # Now scales interface thickness
    
    #Define differential equations
    dndt = (1/q)*dJn - rr_srh - rr_rad - rr_rs - rr_aug       #[m**-3 s**-1]
    dpdt = (-1/q)*dJp - rr_srh - rr_rad - rr_rs - rr_aug      #[m**-3 s**-1]
    
    if do_ss:
        dndt += s.inject_dN
        dpdt += s.inject_dP
    #Package it all together
    dydt = np.hstack([dndt,dpdt])

    return dydt
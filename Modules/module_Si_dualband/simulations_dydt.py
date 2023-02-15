# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:01:07 2021

@author: cfai2
"""
import numpy as np
# from Modules.module_pnJunction.calculations import V_poisson
from calculations import V_poisson

q = 1.0                     # [e]
q_C = 1.602e-19             # [C]
kB = 8.61773e-5             # [eV / K]
eps0 = 8.854e-12 * 1e-9     # [C/V-m] to [C/V-nm]


def dydt_indirect(t, y, G, PA, do_ss=False, verbose=False):
    # Need to tell the function which values of y correspond to n, p, and E
    n_d = y[0:G.nx]
    n_ind = y[G.nx:2*G.nx]
    p = y[2*G.nx:]
    ndp0 = n_d * p - PA.n0 * PA.p0
    n_indp0 = n_ind * p - PA.n0 * PA.p0

    # Define n,p at interior faces
    # Finds average of i and (i+1) - delete last point
    # because (i+1) becomes index 0
    # n_d_faces = ((n_d + np.roll(n_d, -1))/2)[:-1]
    # p_faces = ((p + np.roll(p, -1))/2)[:-1]

    # Flux (current) terms at all faces
    Jn_d = np.zeros(G.nx + 1)
    Jn_ind = np.zeros(G.nx + 1)
    Jp = np.zeros(G.nx + 1)

    # V = V_poisson(G.dx, n_d, p, PA.n0, PA.p0, PA.eps, PA.V0, PA.VL)

    # dVdx = (np.roll(V, -1) - V)[:-1] / G.inter_dx

    # [eV m**-1]
    dEfn_d = (( kB * PA.T * (np.roll(n_d, -1) - n_d)[:-1] / G.inter_dx) -
              0)  # q * n_d_faces * (dVdx + PA.dchidx))

    dEfn_ind = ((kB * PA.T * (np.roll(n_ind, -1) - n_ind)[:-1] / G.inter_dx))

    dEfp = ((-kB * PA.T * (np.roll(p, -1) - p)[:-1] / G.inter_dx) -
            0)   # q * p_faces * (dVdx + PA.dchidx + PA.dEgdx))

    # J at boundaries
    Jn_d[0], Jn_d[G.nx] = 0, 0
    Jn_ind[0], Jn_ind[G.nx] = 0, 0
    Jp[0], Jp[G.nx] = 0, 0

    # Define J at interior faces
    Jn_d[1:-1] = PA.mu_n_d * dEfn_d
    Jn_ind[1:-1] = PA.mu_n_ind * dEfn_ind
    Jp[1:-1] = PA.mu_p * dEfp
    if verbose and t != PA.t_old:
        PA.t_old = t
        print(t)

    # Calculate flux difference
    dJn_d = ((np.roll(Jn_d, -1) - Jn_d)[:-1] / G.dx)  # [(eV/V) m**-3 s**-1]
    dJn_ind = ((np.roll(Jn_ind, -1) - Jn_ind)[:-1] / G.dx)
    dJp = ((np.roll(Jp, -1) - Jp)[:-1] / G.dx)

    rr_srh = (ndp0) / (PA.tauN * p + PA.tauP * n_d)
    rr_rad = PA.B * (ndp0)
    rr_rad_ind = PA.B_ind * (n_indp0)
    rr_aug = (PA.Cn * n_d + PA.Cp * p) * (ndp0)

    rr_rs = np.zeros_like(rr_srh)

    # First "interface" (front surface)
    rr_rs[0] = (PA.RS[0] / G.dx[0]) * (ndp0[0]) / (n_d[0] + p[0])

    # Last "interface" (back surface)
    rr_rs[-1] = (PA.RS[-1] / G.dx[1]) * (ndp0[-1]) / (n_d[-1] + p[-1])

    # Trapping and detrapping from indirect bandgap level

    rr_trap = (n_d - PA.n0) / PA.tauC
    rr_detrap = n_ind / PA.tauE

    # Define transport equations
    dn_d_dt = (1/q) * dJn_d - rr_srh - rr_rad - rr_rs - rr_aug - rr_trap + rr_detrap
    dn_ind_dt = (1/q) * dJn_ind - rr_rad_ind + rr_trap - rr_detrap
    dpdt = (-1/q) * dJp - rr_srh - rr_rad - rr_rad_ind - rr_rs - rr_aug

    if do_ss:
        dn_d_dt += PA.inject_N
        dpdt += PA.inject_P
    # Package it all together
    dydt = np.hstack([dn_d_dt, dn_ind_dt, dpdt])

    return dydt

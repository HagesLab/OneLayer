import numpy as np
from scipy import optimize

from utils import to_pos


def SST(tauN, tauP, n0, p0, B, St, k_fusion, tauT, tauS, tauD_eff, ru_thickness, gen_rate):
    """ Calculates T, S and D densities in steady state conditions."""

    # FIXME: This doesn't work with sequential charge transfer yet
    n_bal = lambda n, src, tn, tp, n0, p0, B: src - (n**2 - n0*p0) * (B + 1/(n * (tn+tp)))
    
    ss_n = optimize.root(n_bal, gen_rate, args=(gen_rate, tauN, tauP, n0, p0, B))
    
    ss_n = ss_n.x # [nm^-3]
    
    T_gen_per_bin = St * (ss_n**2 - n0*p0) / (ss_n + ss_n) / ru_thickness # [nm^-3 ns^-1]
    
    ss_t = 1.05*(-(1/tauT) + np.sqrt(tauT**-2 + 4*k_fusion*T_gen_per_bin)) / (2*k_fusion)
    
    ss_s = k_fusion * tauS * ss_t**2
    ss_d = ss_s * tauD_eff / tauS
    
    return ss_t, ss_s, ss_d


def TTA(T, i, j, need_extra_node, params):
    """
    Calculates TTA rate(x,t) given T data.

    Parameters
    ----------
    T : 1D or 2D ndarray
        Triplet(x,t) values.
    i : int
        Leftmost node index to calculate for.
    j : int
        Rightmost node index to calculate for.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered.
        Most slices involving the index j should include j+1 too
    params : dict {"param name":float or 1D ndarray}
        Collection of parameters from metadata

    Returns
    -------
    PL_base : 2D ndarray
        PL(x,t)

    """
    
    dx = params["Node_width"]
    
    lbound = to_pos(i, dx)
    if need_extra_node:
        ubound = to_pos(j+1, dx)
    else:
        ubound = to_pos(j, dx)
        
    distance = np.arange(lbound, ubound+dx, dx)
    
    if T.ndim == 2: # for integrals of partial thickness
        if need_extra_node:
            T = T[:,i:j+2]
        else:
            T = T[:,i:j+1]
            
        # If the nodes are not equally sized, the need_extra_node and 
        # to_pos mess up and make the distance array one too long. 
        # The user is discouraged from creating such nodes but if they do anyway,
        # Patch here and figure it out later.
        if len(distance) > len(T[0]):
            distance = distance[:len(T[0])]
            
    dT = delta_T({"T":T}, params)
    TTA_base = (params["k_fusion"] * dT ** 2)

    return TTA_base


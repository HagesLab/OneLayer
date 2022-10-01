import numpy as np
from utils import to_pos, to_index, new_integrate
from scipy import integrate as intg

q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]
from scipy.linalg import solve_banded

def V_poisson(N, P, p, dx):
    if N.ndim == 2:
        raise NotImplementedError("Only works with one time step at a time")
    elif N.ndim == 1:
        num_xsteps = len(N)
    
    coef_matrix = np.zeros((3, num_xsteps))
    coef_matrix[1,0] = 1
    coef_matrix[1,-1] = 1
    
    coef_matrix[0, 2:] = 1
    coef_matrix[1, 1:-1] = -2
    coef_matrix[2, :-2] = 1
    
    alpha = -(q_C * dx**2) / (p.eps * eps0) # [V nm^3]
    
    rhs = np.zeros(num_xsteps)
    rhs[:] = alpha * ((P - p.p0) - (N - p.n0))
    rhs[0] = p.V0 # [V]
    rhs[-1] = p.VL
    # rhs[0] = rhs[1]
    # rhs[-1] = rhs[-2]
    return solve_banded((1,1), coef_matrix, rhs)

def E_field(sim_outputs, params):
    """Calculate electric field from N, P"""
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q_C = 1.602e-19 # [C per carrier]
    if isinstance(params["rel_permitivity"], np.ndarray):
        averaged_rel_permitivity = (params["rel_permitivity"][:-1] + np.roll(params["rel_permitivity"], -1)[:-1]) / 2
    else:
        averaged_rel_permitivity = params["rel_permitivity"]
    
    dEdx = q_C * (
        delta_p(sim_outputs, params) - delta_n(sim_outputs, params)
        ) / (eps0 * averaged_rel_permitivity)

    if dEdx.ndim == 1:
        E_field = np.concatenate(([0], np.cumsum(dEdx) * params["Node_width"])) #[V/nm]
        #E_field[-1] = 0

    else:
        E_field = np.concatenate(
            (np.zeros(len(dEdx)).reshape((len(dEdx), 1)),
            np.cumsum(dEdx, axis=1) * params["Node_width"])
            , axis=1) #[V/nm]
        #E_field[:,-1] = 0

    return E_field


def E_field_r(sim_outputs, params):
    """Calculate electric field from T+S+D, Q"""
    eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
    q_C = 1.602e-19 # [C per carrier]
    if isinstance(params["uc_permitivity"], np.ndarray):
        averaged_rel_permitivity = (params["uc_permitivity"][:-1] + np.roll(params["uc_permitivity"], -1)[:-1]) / 2
    else:
        averaged_rel_permitivity = params["uc_permitivity"]
    
    dEdx = q_C * (sim_outputs["P_up"] - (sim_outputs["T"] - params["T0"] + sim_outputs["delta_S"] + sim_outputs["delta_D"])) / (eps0 * averaged_rel_permitivity)
    if dEdx.ndim == 1:
        E_field = np.concatenate(([0], -np.cumsum(dEdx[::-1]) * params["Node_width"]))[::-1] #[V/nm]
        #E_field[-1] = 0
    else:
        E_field = np.concatenate((np.zeros(len(dEdx)).reshape((len(dEdx), 1)), -np.cumsum(dEdx[:, ::-1], axis=1) * params["Node_width"]), axis=1)[:, ::-1] #[V/nm]
        #E_field[:,-1] = 0
    return E_field


def delta_n(sim_outputs, params):
    """Calculate above-equilibrium electron density from N, n0"""
    return sim_outputs["N"] - params["N0"]


def delta_p(sim_outputs, params):
    """Calculate above-equilibrium hole density from P, p0"""
    return sim_outputs["P"] - params["P0"]


def delta_T(sim_outputs, params):
    """Calculate above-equilibrium triplet density from T, T0"""
    return sim_outputs["T"] - params["T0"]


def radiative_recombination(sim_outputs, params):
    """Calculate radiative recombination"""
    return params["B"] * (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"])


def nonradiative_recombination(sim_outputs, params):
    """Calculate nonradiative recombination using SRH model
    Assumes quasi steady state trap level occupation
    """
    return (sim_outputs["N"] * sim_outputs["P"] - params["N0"] * params["P0"]) / ((params["tau_N"] * sim_outputs["P"]) + (params["tau_P"] * sim_outputs["N"]))


def tau_diff(PL, dt):
    """
    Calculates particle lifetime from TRPL.

    Parameters
    ----------
    PL : 1D ndarray
        Time-resolved PL array.
    dt : float
        Time step size.

    Returns
    -------
    1D ndarray
        tau_diff.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        ln_PL = np.where(PL <= 0, 0, np.log(PL))
    
    dln_PLdt = np.zeros(len(ln_PL))
    dln_PLdt[0] = (ln_PL[1] - ln_PL[0]) / dt
    dln_PLdt[-1] = (ln_PL[-1] - ln_PL[-2]) / dt
    dln_PLdt[1:-1] = (np.roll(ln_PL, -1)[1:-1] - np.roll(ln_PL, 1)[1:-1]) / (2*dt)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        dln_PLdt = np.where(dln_PLdt <= 0, 0, -(dln_PLdt ** -1))
    return dln_PLdt

def prep_PL(rad_rec, i, j, need_extra_node, params, layer):
    """
    Calculates PL(x,t) given radiative recombination data plus propogation contributions.

    Parameters
    ----------
    radRec : 1D or 2D ndarray
        Radiative Recombination(x,t) values. These can be MAPI carriers or DBP doublets.
    i : int
        Leftmost node index to calculate for.
    j : int
        Rightmost node index to calculate for.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered.
        Most slices involving the index j should include j+1 too
    params : dict {"param name":float or 1D ndarray}
        Collection of parameters from metadata
    layer : str
        Whether to calculate for layer MAPI or layer DBP.
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
    
    if rad_rec.ndim == 2: # for integrals of partial thickness
        if need_extra_node:
            rad_rec = rad_rec[:,i:j+2]
        else:
            rad_rec = rad_rec[:,i:j+1]
            
        # If the nodes are not equally sized, the need_extra_node and 
        # to_pos mess up and make the distance array one too long. 
        # The user is discouraged from creating such nodes but if they do anyway,
        # Patch here and figure it out later.
        if len(distance) > len(rad_rec[0]):
            distance = distance[:len(rad_rec[0])]
            
    if layer == "MAPI":
        PL_base = (rad_rec)
        
    elif layer == "Rubrene":
        PL_base = rad_rec / params["tau_D"]
    
    return PL_base


class CalculatedOutputs():
    
    def __init__(self, mapi_sim_outputs, ru_sim_outputs, mapi_params, rubrene_params):
        self.mapi_sim_outputs = mapi_sim_outputs
        self.mapi_params = mapi_params
        self.rubrene_sim_outputs = ru_sim_outputs
        self.rubrene_params = rubrene_params
        
    def E_field(self):
        """Calculate electric field from N, P"""
        return E_field(self.mapi_sim_outputs, self.mapi_params)


    def E_field_r(self):
        """Calculate electric field from T+S+D, Q"""
        return E_field_r(self.rubrene_sim_outputs, self.rubrene_params)


    def delta_n(self):
        """Calculate above-equilibrium electron density from N, n0"""
        return delta_n(self.mapi_sim_outputs, self.mapi_params)
    
    def average_delta_n(self, temp_N):
        temp_dN = delta_n({"N":temp_N}, self.mapi_params)
        temp_dN = intg.trapz(temp_dN, dx=self.mapi_params["Node_width"], axis=1)
        temp_dN /= self.mapi_params["Total_length"]
        
        return temp_dN


    def delta_p(self):
        """Calculate above-equilibrium hole density from P, p0"""
        return delta_p(self.mapi_sim_outputs, self.mapi_params)


    def delta_T(self):
        """Calculate above-equilibrium triplet density from T, T0"""
        return delta_T(self.mapi_sim_outputs, self.mapi_params)


    def radiative_recombination(self):
        """Calculate radiative recombination."""
        return radiative_recombination(self.mapi_sim_outputs, self.mapi_params)

    def nonradiative_recombination(self):
        """Calculate nonradiative recombination using SRH model
        Assumes quasi steady state trap level occupation."""
        return nonradiative_recombination(self.mapi_sim_outputs, self.mapi_params)
    
    def mapi_PL(self, temp_N, temp_P):
        temp_RR = radiative_recombination({"N":temp_N, "P":temp_P}, self.mapi_params)
        mapi_thickness = self.mapi_params["Total_length"]
        dm = self.mapi_params["Node_width"]
        PL_base = prep_PL(temp_RR, 0, to_index(mapi_thickness, dm, mapi_thickness), 
                            False, self.mapi_params, "MAPI")
        
        return new_integrate(PL_base, 0, mapi_thickness, dm, mapi_thickness, False)
    
    def dbp_PL(self, temp_D):
        ru_thickness = self.rubrene_params["Total_length"]
        df = self.rubrene_params["Node_width"]
        
        temp_D = prep_PL(temp_D, 0, to_index(ru_thickness, df, ru_thickness), 
                            False, self.rubrene_params, "Rubrene")

        return new_integrate(temp_D, 0, ru_thickness, df, ru_thickness, False)

    def TTA(self, temp_T):
        ru_thickness = self.rubrene_params["Total_length"]
        df = self.rubrene_params["Node_width"]
        temp_T = TTA(temp_T, 0, to_index(ru_thickness, df, ru_thickness), 
                        False, self.rubrene_params)

        return new_integrate(temp_T, 0, ru_thickness, df, ru_thickness, False)
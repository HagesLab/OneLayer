import numpy as np
from utils import to_pos, to_index, new_integrate, to_array, generate_shared_x_array
from scipy import integrate as intg

q = 1.0                     #[e]
q_C = 1.602e-19             #[C]
kB = 8.61773e-5             #[eV / K]
eps0 = 8.854e-12 * 1e-9     #[C/V-m] to [C/V-nm]
from scipy.linalg import solve_banded
def V_poisson_2D(dx, N, P, n0, p0, eps, V0=0, VL=0):
    if N.ndim == 2:
        num_xsteps = len(N[0])
        num_tsteps = len(N)
        V = np.zeros((num_tsteps, num_xsteps))
        for i in range(len(V)): # For each timestep
            V[i] = V_poisson(dx, N[i], P[i], n0, p0, eps, V0, VL)
        return V
    elif N.ndim == 1:
        num_xsteps = len(N)
        V = V_poisson(dx, N, P, n0, p0, eps, V0, VL)
        return V

def V_poisson(dx, N, P, n0, p0, eps, V0=0, VL=0):
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
    
    alpha = -(q_C * dx**2) / (eps * eps0) # [V nm^3]
    
    rhs = np.zeros(num_xsteps)
    rhs[:] = alpha * ((P - p0) - (N - n0))
    rhs[0] = V0 # [V]
    rhs[-1] = VL
    # rhs[0] = rhs[1]
    # rhs[-1] = rhs[-2]
    return solve_banded((1,1), coef_matrix, rhs)

def E_field_from_V(V, dx):
    if V.ndim == 2:
        E = -np.diff(V, axis=1) / dx
        E = np.concatenate([E[:,:1], E, E[:,-1:]], axis=1)
    elif V.ndim == 1:
        E = -np.diff(V) / dx
        E = np.hstack([E[0], E, E[-1]])
    return E

def delta_n(N, n0):
    """Calculate above-equilibrium electron density from N, n0"""
    return N - n0


def delta_p(P, p0):
    """Calculate above-equilibrium hole density from P, p0"""
    return P - p0


def delta_T(sim_outputs, params):
    """Calculate above-equilibrium triplet density from T, T0"""
    return sim_outputs["T"] - params["T0"]


def radiative_recombination(sim_outputs, params):
    """Calculate radiative recombination"""
    return params['B'] * (sim_outputs['N'] * sim_outputs['P'] - params['N0'] * params['P0'])

def nonradiative_recombination(sim_outputs, params):
    """Calculate nonradiative recombination using SRH model
    Assumes quasi steady state trap level occupation
    """
    return (sim_outputs['N'] * sim_outputs['P'] - params['N0'] * params['P0']) / ((params['tau_N'] * sim_outputs['P']) + (params['tau_P'] * sim_outputs['N']))


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
    
    def __init__(self, sim_outputs, params, layer_info):
        self.sim_outputs = sim_outputs
        
        self.layer_names = ["N-type", "buffer", "P-type"]
        total_lengths = [params[layer_name]["Total_length"]
                         for layer_name in self.layer_names]
        
        self.grid_x_edges = [params[layer_name]['edge_x'] for layer_name in self.layer_names]
        self.grid_x_nodes = [params[layer_name]['node_x'] for layer_name in self.layer_names]
        self.dx = np.diff(generate_shared_x_array(True, self.grid_x_edges, total_lengths))
        self.inter_dx = ((self.dx + np.roll(self.dx,-1))/2)[:-1]
        # self.grid_x_edges = generate_shared_x_array(True, self.grid_x_edges, total_lengths)
        # self.grid_x_nodes = generate_shared_x_array(False, self.grid_x_nodes, total_lengths)
        
        self.params = params
        
    def voltage(self):
        """Calculate voltage from N, P"""
        
        get_these_params = ['N0', 'P0', 'rel_permitivity']
        these_params = {}
        
        for p in get_these_params:
            these_params[p] = [to_array(self.params[layer_name][p], len(self.grid_x_nodes[i]),
                                        is_edge=False)
                               for i, layer_name in enumerate(self.layer_names)]
            these_params[p] = generate_shared_x_array(False, these_params[p], total_lengths=None)
        
        return V_poisson_2D(self.dx, self.sim_outputs['N'], self.sim_outputs['P'], 
                         these_params['N0'], these_params['P0'], these_params['rel_permitivity'],
                         V0=0, VL=0)
        
    def E_field(self):
        """Calculate electric field from N, P"""
        V = self.voltage()
        
        return E_field_from_V(V, self.inter_dx)


    def delta_n(self):
        """Calculate above-equilibrium electron density from N, n0"""
        n0 = [to_array(self.params[layer_name]['N0'], len(self.grid_x_nodes[i]),
                       is_edge=False)
              for i, layer_name in enumerate(self.layer_names)]
        n0 = generate_shared_x_array(False, n0, total_lengths=None)
        return delta_n(self.sim_outputs['N'], n0)
    
    def average_delta_n(self, temp_N):
        temp_dN = delta_n({"N":temp_N}, self.mapi_params)
        temp_dN = intg.trapz(temp_dN, dx=self.mapi_params["Node_width"], axis=1)
        temp_dN /= self.mapi_params["Total_length"]
        
        return temp_dN


    def delta_p(self):
        """Calculate above-equilibrium hole density from P, p0"""
        p0 = [to_array(self.params[layer_name]['P0'], len(self.grid_x_nodes[i]),
                       is_edge=False)
              for i, layer_name in enumerate(self.layer_names)]
        p0 = generate_shared_x_array(False, p0, total_lengths=None)
        return delta_n(self.sim_outputs['P'], p0)


    def radiative_recombination(self):
        """Calculate radiative recombination."""
        get_these_params = ['B', 'N0', 'P0']
        these_params = {}
        
        for p in get_these_params:
            these_params[p] = [to_array(self.params[layer_name][p], len(self.grid_x_nodes[i]),
                                        is_edge=False)
                               for i, layer_name in enumerate(self.layer_names)]
            these_params[p] = generate_shared_x_array(False, these_params[p], total_lengths=None)


        return radiative_recombination(self.sim_outputs, these_params)

    def nonradiative_recombination(self):
        """Calculate nonradiative recombination using SRH model
        Assumes quasi steady state trap level occupation."""
        get_these_params = ['N0', 'P0', 'tau_N', 'tau_P']
        these_params = {}
        
        for p in get_these_params:
            these_params[p] = [to_array(self.params[layer_name][p], len(self.grid_x_nodes[i]),
                                        is_edge=False)
                               for i, layer_name in enumerate(self.layer_names)]
            these_params[p] = generate_shared_x_array(False, these_params[p], total_lengths=None)

        return nonradiative_recombination(self.sim_outputs, these_params)
    
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
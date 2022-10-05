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

def prep_PL(rad_rec, i, j, dx, need_extra_node):
    """
    Calculates PL(x,t) given radiative recombination data plus propogation contributions.

    Parameters
    ----------
    radRec : 1D or 2D ndarray
        Radiative Recombination(x,t) values.
    i : int
        Leftmost node index to calculate for.
    j : int
        Rightmost node index to calculate for.
    dx : int
        Node width.
    need_extra_node : bool
        Whether the 'j+1'th node should be considered.
        Most slices involving the index j should include j+1 too
    Returns
    -------
    PL_base : 2D ndarray
        PL(x,t)

    """

    
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
            
    PL_base = (rad_rec)

    
    return PL_base


class CalculatedOutputs():
    
    def __init__(self, sim_outputs, params, layer_info):
        self.sim_outputs = sim_outputs
        
        self.layer_names = ["N-type", "buffer", "P-type"]
        self.total_lengths = [params[layer_name]["Total_length"]
                              for layer_name in self.layer_names]
        
        self.grid_x_edges = [params[layer_name]['edge_x'] for layer_name in self.layer_names]
        self.grid_x_nodes = [params[layer_name]['node_x'] for layer_name in self.layer_names]
        self.dx = np.diff(generate_shared_x_array(True, self.grid_x_edges, self.total_lengths))
        self.inter_dx = ((self.dx + np.roll(self.dx,-1))/2)[:-1]
        # self.grid_x_edges = generate_shared_x_array(True, self.grid_x_edges, total_lengths)
        # self.grid_x_nodes = generate_shared_x_array(False, self.grid_x_nodes, total_lengths)
        
        self.params = params
        
    def get_stitched_params(self, get_these_params):
        these_params = {}
        
        for p in get_these_params:
            these_params[p] = [to_array(self.params[layer_name][p], len(self.grid_x_nodes[i]),
                                        is_edge=False)
                               for i, layer_name in enumerate(self.layer_names)]
            these_params[p] = generate_shared_x_array(False, these_params[p], total_lengths=None)
        
        return these_params
    
    def voltage(self):
        """Calculate voltage from N, P"""
        get_these_params = ['N0', 'P0', 'rel_permitivity']
        these_params = self.get_stitched_params(get_these_params)
        
        return V_poisson_2D(self.dx, self.sim_outputs['N'], self.sim_outputs['P'], 
                         these_params['N0'], these_params['P0'], these_params['rel_permitivity'],
                         V0=0, VL=0)
        
    def E_field(self):
        """Calculate electric field from N, P"""
        V = self.voltage()
        
        return E_field_from_V(V, self.inter_dx)


    def delta_n(self):
        """Calculate above-equilibrium electron density from N, n0"""
        get_these_params = ['N0']
        these_params = self.get_stitched_params(get_these_params)

        return delta_n(self.sim_outputs['N'], these_params['N0'])
    
    def average_delta_n(self, temp_N):
        get_these_params = ['N0']
        these_params = self.get_stitched_params(get_these_params)
        node_list = generate_shared_x_array(False, self.grid_x_nodes, self.total_lengths)
        
        temp_dN = delta_n(temp_N, these_params['N0'])
        temp_dN = intg.trapz(temp_dN, x=node_list, axis=1)
        temp_dN /= sum(self.total_lengths)
        
        return temp_dN


    def delta_p(self):
        """Calculate above-equilibrium hole density from P, p0"""
        get_these_params = ['P0']
        these_params = self.get_stitched_params(get_these_params)
        return delta_p(self.sim_outputs['P'], these_params['P0'])


    def radiative_recombination(self):
        """Calculate radiative recombination."""
        get_these_params = ['B', 'N0', 'P0']
        these_params = self.get_stitched_params(get_these_params)

        return radiative_recombination(self.sim_outputs, these_params)

    def nonradiative_recombination(self):
        """Calculate nonradiative recombination using SRH model
        Assumes quasi steady state trap level occupation."""
        get_these_params = ['N0', 'P0', 'tau_N', 'tau_P']
        these_params = self.get_stitched_params(get_these_params)
        
        return nonradiative_recombination(self.sim_outputs, these_params)
    
    def PL(self, temp_N, temp_P):
        get_these_params = ['B', 'N0', 'P0']
        these_params = self.get_stitched_params(get_these_params)
        
        temp_RR = radiative_recombination({"N":temp_N, "P":temp_P}, these_params)
        
        # Integrate PL separately for each layer
        # TODO: This for all layers at once using the variable dx grid,
        # so we don't have to awkwardly break RR into individual layer contributions
        # by calculating indices l and r
        l = 0
        r = 0
        integral_PL = np.zeros_like(temp_RR[:, 0])
        for i in range(len(self.layer_names)):
            r += len(self.grid_x_nodes[i])
            thickness = self.total_lengths[i]
            dx_len = self.grid_x_edges[i][1] - self.grid_x_edges[i][0]
            PL_base = prep_PL(temp_RR[:, l:r+1], 0, to_index(thickness, dx_len, thickness),
                              dx_len, False)
            l = r
            integral_PL += new_integrate(PL_base, 0, thickness, dx_len, thickness, False)
            
        return integral_PL
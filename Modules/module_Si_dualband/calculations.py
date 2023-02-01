import numpy as np
from scipy.linalg import solve_banded

q = 1.0                     # [e]
q_C = 1.602e-19             # [C]
kB = 8.61773e-5             # [eV / K]
eps0 = 8.854e-12 * 1e-9     # [C/V-m] to [C/V-nm]


def V_poisson_2D(dx, N, P, n0, p0, eps, V0=0, VL=0):
    if N.ndim == 2:
        num_xsteps = len(N[0])
        num_tsteps = len(N)
        V = np.zeros((num_tsteps, num_xsteps))
        for i in range(len(V)):  # For each timestep
            V[i] = V_poisson(dx, N[i], P[i], n0, p0, eps, V0, VL, mode='ocv')
        return V
    elif N.ndim == 1:
        num_xsteps = len(N)
        V = V_poisson(dx, N, P, n0, p0, eps, V0, VL, mode='ocv')
        return V


def V_poisson(dx, N, P, n0, p0, eps, k0=0, kL=0, mode='ocv'):
    if N.ndim == 2:
        raise NotImplementedError("Only works with one time step at a time")
    elif N.ndim == 1:
        num_xsteps = len(N)

    alpha = -(q_C * dx**2) / (eps * eps0)  # [V nm^3]
    rhs = np.zeros(num_xsteps)
    rhs[:] = alpha * ((P - p0) - (N - n0))

    coef_matrix = np.zeros((3, num_xsteps))

    coef_matrix[0, 2:] = 1
    coef_matrix[1, 1:-1] = -2
    coef_matrix[2, :-2] = 1

    if mode == 'fv':
        raise NotImplementedError("No fixed voltage BC yet")
        # Fixed voltage #####
        # These set the boundary equations V[0] = V0, V[L] = VL
        # coef_matrix[1,0] = 1
        # coef_matrix[0,1] = 0
        # coef_matrix[1,-1] = 1
        # coef_matrix[2,-2] = 0

        # # These set the actual values of V0 and VL
        # rhs[0] = p.V0 # [V]
        # rhs[-1] = p.VL
        ####################
    elif mode == 'ocv':
        # Open circuit
        # Where k0 is the fixed voltage [V] at left bound and kL is the fixed
        # Efield [V/nm] at right bound
        coef_matrix[1, 0] = -3
        coef_matrix[0, 1] = 1
        coef_matrix[1, -1] = -1
        coef_matrix[2, -2] = 1
        # These set the values of k0 and kL
        rhs[0] = rhs[0] - 2 * k0
        rhs[-1] = rhs[-1] + kL * dx[-1]

    return solve_banded((1, 1), coef_matrix, rhs)


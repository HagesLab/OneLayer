# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:53:38 2021

@author: cfai2
"""
import numpy as np
from scipy import integrate as intg

def to_index(x,dx, absUpperBound, is_edge=False):
    """Returns largest node index less than or equal to position x
    # Warning: this always rounds x down to the nearest node (or edge if is_edge=True)!"""
    absLowerBound = dx / 2 if not is_edge else 0
    
    if (x < 0):
        raise ValueError("cannot index negative position")
    if (x < absLowerBound):
        return 0

    if (x > absUpperBound):
        raise ValueError("position larger than length of system")

    return int((x - absLowerBound) / dx)

def to_pos(i,dx, is_edge=False):
    """ Returns position x corresponding to node index i"""
    absLowerBound = dx / 2 if not is_edge else 0
    return (absLowerBound + i * dx)

def to_array(value, m, is_edge):
    """Casts value to uniform 1D ndarray if necessary"""
    if not isinstance(value, np.ndarray):
        if is_edge:
            return np.ones(m+1) * value
        else:
            return np.ones(m) * value
        
    else:
        return value
    
def new_integrate(base_data, l_bound, u_bound, dx, total_length, need_extra_node):
    """
    General purpose integration function using scipy.trapz()
    Integrates over 2nd dimension.

    Parameters
    ----------
    base_data : 1D or 2D ndarray
        Values to integrate over.
    l_bound : float
        Lower boundary.
    u_bound : float
        Upper boundary.
    dx : float
        Space node width.
    total_length : float
        Length of system
    need_extra_node : bool
        Whether the smallest node with position larger than u_bound should be considered.
        Correction for converting from actual boundaries to discrete nodes

    Returns
    -------
    I_data : 1D array
        Integrated values.

    """
    i = to_index(l_bound, dx, total_length)
    j = to_index(u_bound, dx, total_length)
    if base_data.ndim == 1:
        base_data = base_data[None]
    
    if l_bound == u_bound:
        I_base = base_data[:,0]
        if l_bound >= to_pos(i, dx) + dx / 2 and not l_bound == total_length:
            I_plus_one = base_data[:,1]

        if l_bound == to_pos(i, dx) + dx / 2 and not l_bound == total_length:
            I_data = (I_base + I_plus_one) / 2

        elif l_bound > to_pos(i, dx) + dx / 2:
            I_data = I_plus_one

        else:
            I_data = I_base
    else:
        if need_extra_node:
            I_base = base_data
            I_data = intg.trapz(I_base[:, :-1], dx=dx, axis=1)
            
        else:
            I_base = base_data
            I_data = intg.trapz(I_base, dx=dx, axis=1)

        I_data += correct_integral(I_base.T, l_bound, u_bound, i, j, dx)
    return I_data

def correct_integral(integrand, l_bound, u_bound, i, j, dx):
    """
    Corrects new_integrate() for mismatch between nodes and actual integration bounds using linear interpolation.

    Parameters
    ----------
    integrand : 1D ndarray
        Raw result of new_integrate()
    l_bound : float
        Lower boundary.
    u_bound : float
        Upper boundary.
    i : int
        Index of lowest node integrated over.
    j : int
        Index of highest node integrated over.
        Note that l_bound, u_bound do not correspond exactly with i,j hence the correction is needed
    dx : float
        Space node width.

    Returns
    -------
    1D ndarray
        Corrected integration results.

    """
    uncorrected_l_bound = to_pos(i, dx)
    uncorrected_u_bound = to_pos(j, dx)
    lfrac1 = min(l_bound - uncorrected_l_bound, dx / 2)

    # Yes, integrand[0] and not integrand[i]. Note that in integrate(), the ith node maps to integrand[0] and the jth node maps to integrand[j-i].
    l_bound_correction = integrand[0] * lfrac1

    if l_bound > uncorrected_l_bound + dx / 2:
        lfrac2 = (l_bound - (uncorrected_l_bound + dx / 2))
        l_bound_correction += integrand[0+1] * lfrac2

    ufrac1 = min(u_bound - uncorrected_u_bound, dx / 2)
    u_bound_correction = integrand[j-i] * ufrac1
    
    if u_bound > uncorrected_u_bound + dx / 2:
        ufrac2 = (u_bound - (uncorrected_u_bound + dx / 2))
        try:
            u_bound_correction += integrand[j-i+1] * ufrac2
        except Exception:
            u_bound_correction += integrand[j-i] * ufrac2
            print("An index mismatch occured while calculating u_bound_correction\n"
                  "Values may differ slightly than expected")

    return u_bound_correction - l_bound_correction
    
def get_all_combinations(value_dict):
    """
    >>> value_dict = {"a":[1,2], "b":[3,4]}
    
    >>> g = get_all_combinations(value_dict)
    
    >>> g
    
    [{"a":1, "b":3}, {"a":1, "b":4},
     {"a":2, "b":3}, {"a":2, "b":4}]
    """
    
    combinations = []
    param_names = list(value_dict.keys())
        
    iterable_param_indexes = {}
    iterable_param_lengths = {}
    for param in param_names:
        iterable_param_indexes[param] = 0
        iterable_param_lengths[param] = len(value_dict[param])
    
    pivot_index = len(param_names) - 1

    current_params = dict(value_dict)
    # Create a list of all combinations of parameter values
    while(pivot_index >= 0):

        # Generate the next parameter set using lists of indices stored in the helper structures
        for iterable_param in param_names:
            current_params[iterable_param] = value_dict[iterable_param][iterable_param_indexes[iterable_param]]

        combinations.append(dict(current_params))

        # Determine the next iterable parameter using a "reverse search" amd update indices from right to left
        # For example, given Param_A = [1,2,3], Param_B = [4,5,6], Param_C = [7,8]:
        # The order {A, B, C} this algorithm will run is: 
        # {1,4,7}, 
        # {1,4,8}, 
        # {1,5,7}, 
        # {1,5,8}, 
        # {1,6,7}, 
        # {1,6,8},
        # ...
        # {3,6,7},
        # {3,6,8}
        pivot_index = len(param_names) - 1
        while (pivot_index >= 0 and iterable_param_indexes[param_names[pivot_index]] == iterable_param_lengths[param_names[pivot_index]] - 1):
            pivot_index -= 1

        iterable_param_indexes[param_names[pivot_index]] += 1

        for i in range(pivot_index + 1, len(param_names)):
            iterable_param_indexes[param_names[i]] = 0
            
    return combinations
    
    
def autoscale(val_array=None, min_val=None, max_val=None):
    """Help a matplotlib plot determine whether a log or linear scale should be used
       when plotting val_array
    """
    if max_val is not None and min_val is not None:
        pass
    elif val_array is not None:
        max_val = np.amax(val_array)
        min_val = np.amin(val_array)
        
    else:
        return 'linear'
    try:
        if not (min_val == 0) and np.abs(max_val / min_val) > 1e1:
            return 'symlog'
        elif (min_val < 0) and np.abs(min_val / max_val) > 1e1:
            return 'symlog'
        else:
            return 'linear'
    except Exception:
        return 'linear'
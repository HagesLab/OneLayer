#################################################
# Functions for common device excitation profiles
################################################# 

import numpy as np

def pulse_laser_power_spotsize(power, spotsize, freq, wavelength, 
                               alpha, x_array, hc=6.626e-34*2.997e8):
    """
    Initial excitation profile.
    Parameters
    ----------
    power : float
        Laser power.
    spotsize : float
        Laser beam cross section.
    freq : float
        Laser pulse frequency.
    wavelength : float
        Laser light wavelength.
    alpha : float
        Attenuation coefficient.
    x_array : 1D numpy array
        Space grid
    hc : float, optional
        h and c are Planck's const and speed of light, respectively. These default to common units [J*s] and [m/s] but
        they may be passed in with different units. 
        The default is 6.626e-34*2.997e8 [J*m].

    Returns
    -------
    1D ndarray
        Excited carrier profile.

    """
    
    return (power / (spotsize * freq * hc / wavelength) * alpha * np.exp(-alpha * x_array))

def pulse_laser_powerdensity(power_density, freq, wavelength, 
                             alpha, x_array, hc=6.626e-34*2.997e8):
    """
    Initial excitation profile.
    Parameters
    ----------
    power_density : float
        Laser power density.
    
    See pulse_laser_power_spotsize for more details.

    """
    return (power_density / (freq * hc / wavelength) * alpha * np.exp(-alpha * x_array))

def pulse_laser_maxgen(max_gen, alpha, x_array, hc=6.626e-34*2.997e8):
    """
    Initial excitation profile.
    Parameters
    ----------
    max_gen : float
        Maximum carrier density of excitation profile.
    See pulse_laser_power_spotsize for more details.

    """
    return (max_gen * np.exp(-alpha * x_array))

def pulse_laser_totalgen(total_gen, total_length, alpha, x_array, hc=6.626e-34*2.997e8):
    """
    Initial excitation profile
    Parameters
    ----------
    total_gen : float
        Total excited carrier density.
    total_length : float
        Length of system.
    See pulse_laser_power_spotsize for more details.

    """
    return ((total_gen * total_length * alpha * np.exp(alpha * total_length)) / (np.exp(alpha * total_length) - 1) * np.exp(-alpha * x_array))
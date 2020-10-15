import numpy as np


def readin_densprofile():
    PATH = '/Users/jonpetersen/data/data_BA/'
    MSIS_DATEI = 'MSIS/MSIS_18072300_new.txt'
    with open(PATH + MSIS_DATEI) as msis:
        MSISdata = np.genfromtxt(msis, skip_header=11)
    # 0 Height, km | 1 O, cm-3 | 2 N2, cm-3 | 3 O2, cm-3 |
    # 4 Mass_density, g/cm-3 | 5 Ar, cm-3
    MSISalt = MSISdata[:,0]        # Altitude
    MSISdens = (MSISdata[:,1] + MSISdata[:,2] + MSISdata[:,3] +
        MSISdata[:,4]) * 10**6 # add desitys and convert to SI units
    return MSISdens, MSISalt


def phasefunc(ang, g = 0.7):
    """Phasefunction after Henyey Greenstein.
    g must be between -1 (full backscatter) and 1 (forwardscattering)."""

    if type(ang) not in [int, float]:
        raise TypeError('The angle must be a real number between 0 and 180')
    if ang < 0 and ang > 180:
        raise ValueError('The angle must be positive and below 180')
    if type(g) not in [int, float]:
        raise TypeError('g be a real number between -1 and 1')
    if g < -1 and g > 1:
        raise ValueError('g must be between -1 and 1')

    return (1 - g**2) / (1 + g**2 - 2 * g * np.cos(ang) )**(3/2)


def calc_scattering_angle(theta_in, theta_out, phi_in, phi_out):
    """Calculates the angle between the incoming and outgoing pencilbeam.
    It is needed for the Phasefunction. Returns the angle in deg.
    After Stamnes eqn. 3.22.
    """
    if type(theta_in) not in [int, float]:
        raise TypeError('Theta in must be a real number between 0 and 180')
    if theta_in < 0 and theta_in > 180:
        raise ValueError('Theta in cannot be negative or greater 180')
    if type(theta_out) not in [int, float]:
        raise TypeError('Theta out must be a real number between 0 and 180')
    if theta_out < 0 and theta_out > 180:
        raise ValueError('Theta out cannot be negative or greater 180')
    if type(phi_in) not in [int, float]:
        raise TypeError('Phi in must be a real number between 0 and 360')
    if phi_in < 0 and phi_in >= 360:
        raise ValueError('Phi cannot be negative or >= 360')
    if type(phi_out) not in [int, float]:
        raise TypeError('Phi out must be a real number between 0 and 360')
    if phi_out < 0 and phi_out >= 360:
        raise ValueError('Phi out cannot be negative or >= 360')

    angle = np.cos(RT_model_1D.deg2rad(theta_out)) * \
            np.cos(RT_model_1D.deg2rad(theta_in)) + \
            np.sin(RT_model_1D.deg2rad(theta_out)) * \
            np.sin(RT_model_1D.deg2rad(theta_in)) * \
            np.cos(RT_model_1D.deg2rad(phi_in - phi_out))

    return float( RT_model_1D.rad2deg( np.arccos(angle) ))


def argclosest(value, array, return_value = False):
    """Returns the index in ``array`` which is closest to ``value``."""
    idx = np.abs(array - value).argmin()
    return (idx, array[idx].item()) if return_value else idx


def delta_func(n):
    """ Delta function """
    if type(n) not in [int, float]:
        raise TypeError('Only numbers can be an input!')

    if n == 0:
        return 1
    else:
        return 0


"""
## Converting Units
"""
def deg2rad(deg):
    """Converts degress to radian"""
    return deg * np.pi / 180


def rad2deg(rad):
    """Converts degress to radian"""
    return rad * 180 / np.pi


def km2m(km):
    """Converts km to m"""
    return km * 1000


def m2km(m):
    """Converts m to km"""
    return m / 1000

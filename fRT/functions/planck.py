import numpy as np
from fRT import constants

__all__ = [
    "planck_freq",
    "planck_wavelength",
    "sun_init_intensity"
    ]


def planck_freq(freq, temp, stokes_dim=1):
    """Returns the intensity of electromagnetic radiation at the frequency
    (freq) emitted by a black body in thermal equilibrium at a given
    temperature (temp).
    """
    c_0 = constants.speed_of_light
    h = constants.planck
    kb = constants.boltzmann

    factor = np.exp((h * freq) / (kb * temp))
    B = 2 * h * freq ** 3 / c_0 ** 2 * 1 / (factor - 1)

    stokes = np.zeros(stokes_dim)
    stokes[0] = B

    return B if stokes_dim == 1 else stokes


def planck_wavelength(lam, temp, stokes_dim=1):
    """Returns the intensity of electromagnetic radiation at the wavelength
    (lam) emitted by a black body in thermal equilibrium at a given
    temperature (temp).
    """
    c_0 = constants.speed_of_light
    h = constants.planck
    kb = constants.boltzmann

    factor = np.exp((h * c_0) / (lam * kb * temp))
    B = 2 * h * c_0 ** 2 / lam ** 5 * 1 / (factor - 1)

    stokes = np.zeros(stokes_dim)
    stokes[0] = B

    return B if stokes_dim == 1 else stokes

def sun_init_intensity(wavelength, stokes_dim=1):
    """ Returns the sun intensity for the field initialisation. """
    sun_eff_temp = constants.sun_eff_temp
    sun_solid_angle = constants.sun_solid_angle
    rad = planck_wavelength(wavelength, sun_eff_temp)
    rad *= sun_solid_angle / (4*np.pi) # normalize
    stokes = np.zeros(stokes_dim)
    stokes[0] = rad

    return rad if stokes_dim == 1 else stokes

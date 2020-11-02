import numpy as np
from fRT import constants

__all__ = [
    "planck_freq",
    "planck_wavelength"
    ]


def planck_freq(freq, temp):
    """Returns the intensity of electromagnetic radiation at the frequency
    (freq) emitted by a black body in thermal equilibrium at a given
    temperature (temp).
    """
    c_0 = constants.speed_of_light
    h = constants.planck
    kb = constants.boltzmann

    factor = np.exp((h * freq) / (kb * temp))
    B = 2 * h * freq ** 3 / c_0 ** 2 * 1 / (factor - 1)

    return B


def planck_wavelength(lam, temp):
    """Returns the intensity of electromagnetic radiation at the wavelength
    (lam) emitted by a black body in thermal equilibrium at a given
    temperature (temp).
    """
    c_0 = constants.speed_of_light
    h = constants.planck
    kb = constants.boltzmann

    factor = np.exp((h * c_0) / (lam * kb * temp))
    B = 2 * h * c_0 ** 2 / lam ** 5 * 1 / (factor - 1)

    return B

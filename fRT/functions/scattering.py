import numpy as np

__all__ = [
    "henyey_greenstein_phasefunc",
    "rayleigh_phasefunc",
    "rayleigh_phasematrix",
    "calc_scattering_angle",
    "calc_rayleigh_scattering_cross_section",
]


def henyey_greenstein_phasefunc(ang, g=0.7):
    """Phasefunction after Henyey Greenstein. The anisotropy factor g
    must be between -1 (full backscatter) and 1 (forwardscattering).
    The Phasefunction is normalized with 4 Pi."""

    if ang < 0 and ang > 180:
        raise ValueError("The angle must be positive and below 180")
    if g < -1 and g > 1:
        raise ValueError("g must be between -1 and 1")

    P = (1 - g ** 2) / (1 + g ** 2 - 2 * g * np.cos(np.deg2rad(ang))) ** (3 / 2)
    return P / (4 * np.pi)


def rayleigh_phasefunc(ang):
    """Phasefunction for rayleigh scattering.
    The Phasefunction is normalized with 4 Pi."""
    if ang < 0 and ang > 180:
        raise ValueError("The angle must be positive and below 180")

    P = 3 / 4 * (1 + np.cos(np.deg2rad(ang)) ** 2)
    return P / (4 * np.pi)


def rayleigh_phasematrix(ang, stokes_dim=4):
    """Phasematrix for rayleigh scattering.
    The Phasematrix is normalized with 4 Pi.
    Based on Liou eqn in exercise 5.9"""
    if ang < 0 and ang > 180:
        raise ValueError("The angle must be positive and below 180")

    a11 = a22 = 1 / 2 * (1 + np.cos(np.deg2rad(ang)) ** 2)
    a21 = a12 = -1 / 2 * np.sin(np.deg2rad(ang))
    a33 = a44 = np.cos(np.deg2rad(ang))
    P = np.zeros((4, 4))
    P[0, 0], P[1, 1], P[2, 2], P[3, 3] = a11, a22, a33, a44
    P[1, 0], P[0, 1] = a21, a12
    return 3 / 2 * P[:stokes_dim, :stokes_dim] / (4 * np.pi)


def calc_scattering_angle(theta_in, theta_out, phi_in, phi_out):
    """Calculates the angle between the incoming and outgoing pencilbeam.
    It is needed for the Phasefunction. Returns the angle in deg.
    After Stamnes eqn. 3.22.
    """
    if theta_in < 0 and theta_in > 180:
        raise ValueError("Theta in cannot be negative or greater 180")
    if theta_out < 0 and theta_out > 180:
        raise ValueError("Theta out cannot be negative or greater 180")
    if phi_in < 0 and phi_in >= 360:
        raise ValueError("Phi cannot be negative or >= 360")
    if phi_out < 0 and phi_out >= 360:
        raise ValueError("Phi out cannot be negative or >= 360")

    angle = np.cos(np.deg2rad(theta_out)) * np.cos(np.deg2rad(theta_in)) + np.sin(
        np.deg2rad(theta_out)
    ) * np.sin(np.deg2rad(theta_in)) * np.cos(np.deg2rad(phi_in - phi_out))
    if angle > 1:
        angle = 1.0
    return float(np.rad2deg(np.arccos(angle)))


def calc_rayleigh_scattering_cross_section(wavelength):
    """Calculates the scattering cross section for rayleigh scattering.
    A numerical approximation (accurate to 0.3%) for the Rayleigh scattering
    cross section for air from Stamnes Chapter 3.3.7, the eqn after 3.21.
    The approximation is valid for 0.205 < lambda < 1.05 micrometers.
    (This formula was provided by M. Callan, University of Colorado, who fitted
    the numerical results of Bates (1984).)
    """
    if wavelength < 0:
        raise ValueError("The wavelengh cannot be negative")
    wavelength = wavelength * 1e6
    constants = [3.9729066, 4.6547659e-2, 4.5055995e-4, 2.3229848e-5]
    sum = 0
    for i in range(4):
        sum += constants[i] * wavelength ** (-2 * i)
    sigma_ray = wavelength ** (-4) * sum * 1e-28
    return sigma_ray * 1e-4  # converted into m^

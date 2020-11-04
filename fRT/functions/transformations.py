import numpy as np
from fRT.functions.scattering import *

__all__ = [
    "stokes_rotation_matrix",
    "transformation_angle",
    "transformed_rayleigh_scattering_matrix",
]


def stokes_rotation_matrix(eta):
    """Calculates the Stokes rotation matrix L(eta) for the angle eta.

    Based on Mishchenko eqn 1.97."""
    # if eta < 0 and eta > 180:
    #     raise ValueError('The angle must be positive and below 180')
    L = np.zeros((4, 4))
    L[0, 0], L[4, 4] = 1, 1
    L[1, 1], L[2, 2] = np.cos(np.deg2rad(2 * eta)), np.cos(np.deg2rad(2 * eta))
    L[2, 1], L[1, 2] = np.sin(np.deg2rad(2 * eta)), -np.sin(np.deg2rad(2 * eta))
    return L


def transformation_angle(theta_in, theta_out, phi_in, phi_out):
    """Calculates the transformation angles for the stokes_rotation_matrix.

    Based on Mishchenko eqn 4.18 and 4.19.
    """

    theta_sca = calc_scattering_angle(theta_in, theta_out, phi_in, phi_out)

    sigma1 = np.argcos(
        (
            np.cos(np.deg2rad(theta_out))
            - np.cos(np.deg2rad(theta_in)) * np.cos(np.deg2rad(theta_sca))
        )
        / np.sin(np.deg2rad(theta_in))
        * np.sin(np.deg2rad(theta_sca))
    )
    sigma2 = np.argcos(
        (
            np.cos(np.deg2rad(theta_in))
            - np.cos(np.deg2rad(theta_out)) * np.cos(np.deg2rad(theta_sca))
        )
        / np.sin(np.deg2rad(theta_out))
        * np.sin(np.deg2rad(theta_sca))
    )
    return sigma1, sigma2


def transformed_rayleigh_scattering_matrix(
    theta_in, theta_out, phi_in, phi_out, stokes_dim=4
):
    """The scattering matrix for the transport coordinate system.

    Based on Mishchenko eqn 4.14.
    """
    if theta_in < 0 and theta_in > 180:
        raise ValueError("Theta in cannot be negative or greater 180")
    if theta_out < 0 and theta_out > 180:
        raise ValueError("Theta out cannot be negative or greater 180")
    if phi_in < 0 and phi_in >= 360:
        raise ValueError("Phi cannot be negative or >= 360")
    if phi_out < 0 and phi_out >= 360:
        raise ValueError("Phi out cannot be negative or >= 360")

    theta_sca = calc_scattering_angle(theta_in, theta_out, phi_in, phi_out)
    sigma1, sigma2 = transformation_angle(theta_in, theta_out, phi_in, phi_out)

    L1 = stokes_rotation_matrix(-sigma2)
    L2 = stokes_rotation_matrix(180 - sigma1)
    F = rayleigh_phasematrix(theta_sca)

    P = L1 @ F @ L2
    return P[:stokes_dim, :stokes_dim]

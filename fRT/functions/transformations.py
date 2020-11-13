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
    L[0, 0], L[3, 3] = 1, 1
    L[1, 1], L[2, 2] = np.cos(np.deg2rad(2 * eta)), np.cos(np.deg2rad(2 * eta))
    L[2, 1], L[1, 2] = np.sin(np.deg2rad(2 * eta)), -np.sin(np.deg2rad(2 * eta))
    return L


def transformation_angle(theta_in, theta_out, phi_in, phi_out):
    """Calculates the transformation angles for the stokes_rotation_matrix.

    Based on Mishchenko eqn 4.18 and 4.19.
    """

    theta_sca = calc_scattering_angle(theta_in, theta_out, phi_in, phi_out)

    sigma1 = np.arccos(
        (
            np.cos(np.deg2rad(theta_out))
            - np.cos(np.deg2rad(theta_in)) * np.cos(np.deg2rad(theta_sca))
        )
        / (np.sin(np.deg2rad(theta_in))
        * np.sin(np.deg2rad(theta_sca)))
    )
    sigma2 = np.arccos(
        (
            np.cos(np.deg2rad(theta_in))
            - np.cos(np.deg2rad(theta_out)) * np.cos(np.deg2rad(theta_sca))
        )
        / (np.sin(np.deg2rad(theta_out))
        * np.sin(np.deg2rad(theta_sca)))
    )
    return np.rad2deg(sigma1), np.rad2deg(sigma2)


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
    F = rayleigh_phasematrix(theta_sca)

    P = np.zeros((stokes_dim,stokes_dim))

    # for stokes dim = 1, we only need Z11 = F11
    P[0,0] = F[0, 0]
    if stokes_dim == 1:
        return P.squeeze()

    #
    # Multiple cases have to be considered:
    #
    ANGTOL_RAD = 1e-6 # absolut tolerenz for angle
    ANGTOL = np.rad2deg(ANGTOL_RAD)

    if (
            abs(theta_sca) < ANGTOL             # forward scattering
            or abs(theta_sca - 180) < ANGTOL   # backward scattering
            or abs(phi_in - phi_out) < ANGTOL  # inc and sca on meridian
            or abs(abs(phi_in - phi_out) - 360.) < ANGTOL
            or abs(abs(phi_in - phi_out) - 180.) < ANGTOL):

        P[0,1], P[1,0], P[1,1] = F[0,1], F[1,0], F[1,1]
        if stokes_dim > 2:
            P[2,2] = F[2,2]
            # other elements are 0
            if stokes_dim > 3:
                P[2,3], P[3,2], P[3,3] = F[2,3], F[3,2], F[3,3]
                # other elements are 0
    else:
        if theta_in < ANGTOL:
            sigma1 = 180 + phi_out - phi_in
            sigma2 = 0
        elif theta_in > 180 - ANGTOL:
            sigma1 = phi_out - phi_in
            sigma2 = 180
        elif theta_out < ANGTOL:
            sigma1 = 0
            sigma2 = 180 + phi_out - phi_in
        elif theta_out > 180 - ANGTOL:
            sigma1 = 180
            sigma2 = phi_out - phi_in
        else:
            sigma1, sigma2 = transformation_angle(theta_in, theta_out, phi_in, phi_out)

        L1 = stokes_rotation_matrix(180 - sigma1)
        L2 = stokes_rotation_matrix(-sigma2)

        P = L2 @ F @ L1

    return P[:stokes_dim, :stokes_dim]

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
    return MSISalt, MSISdens


def readin_tempprofile():
    PATH = '/Users/jonpetersen/data/data_BA/'
    FILE = 'T_Fit_54N_Tab.txt'
    with open(PATH + FILE) as f:
        data = np.genfromtxt(f, skip_header = 11, usecols = None)

    altitude = np.empty((len(data)))
    mean_temp = np.empty((len(data)))
    for id, row in enumerate(data):
        altitude[id] = row[0]
        mean_temp[id] = np.mean(row[1:])

    return altitude, mean_temp


def henyey_greenstein_phasefunc(ang, g = 0.7):
    """Phasefunction after Henyey Greenstein. The anisotropy factor g
    must be between -1 (full backscatter) and 1 (forwardscattering).
    The Phasefunction is normalized with 4 Pi."""

    if type(ang) not in [int, float]:
        raise TypeError('The angle must be a real number between 0 and 180')
    if ang < 0 and ang > 180:
        raise ValueError('The angle must be positive and below 180')
    if type(g) not in [int, float]:
        raise TypeError('g be a real number between -1 and 1')
    if g < -1 and g > 1:
        raise ValueError('g must be between -1 and 1')

    P = (1 - g**2) / (1 + g**2 - 2 * g * np.cos(np.deg2rad(ang)) )**(3/2)
    return P / (4 * np.pi)


def rayleigh_phasefunc(ang):
    """Phasefunction for rayleigh scattering.
    The Phasefunction is normalized with 4 Pi."""
    if type(ang) not in [int, float]:
        raise TypeError('The angle must be a real number between 0 and 180')
    if ang < 0 and ang > 180:
        raise ValueError('The angle must be positive and below 180')

    P = 3/4 * (1 + np.cos(np.deg2rad(ang))**2)
    return P / (4 * np.pi)


def rayleigh_phasematrix(ang, stokes_dim = 4):
    """Phasematrix for rayleigh scattering.
    The Phasematrix is normalized with 4 Pi.
    Based on Liou eqn in exercise 5.9"""
    if type(ang) not in [int, float]:
        raise TypeError('The angle must be a real number between 0 and 180')
    if ang < 0 and ang > 180:
        raise ValueError('The angle must be positive and below 180')

    a11 = a22 = 1/2 * (1 + np.cos(np.deg2rad(ang))**2)
    a21 = a12 = -1/2 * np.sin(np.deg2rad(ang))
    a33 = a44 = np.cos(np.deg2rad(ang))
    P = np.zeros((4,4))
    P[0,0], P[1,1], P[2,2], P[3,3] = a11, a22, a33, a44
    P[1,0], P[0,1] = a21, a12
    return 3/2 * P[:stokes_dim,:stokes_dim] / (4 * np.pi)


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

    angle = np.cos(np.deg2rad(theta_out)) * \
            np.cos(np.deg2rad(theta_in)) + \
            np.sin(np.deg2rad(theta_out)) * \
            np.sin(np.deg2rad(theta_in)) * \
            np.cos(np.deg2rad(phi_in - phi_out))

    return float(np.rad2deg(np.arccos(angle)))


def stokes_rotation_matrix(eta):
    """ Calculates the Stokes rotation matrix L(eta) for the angle eta.
    Based on Mishchenko eqn 1.97."""
    if type(eta) not in [int, float]:
        raise TypeError('The angle must be a real number')
    # if eta < 0 and eta > 180:
    #     raise ValueError('The angle must be positive and below 180')
    L = np.zeros((4,4))
    L[0,0], L[4,4] = 1, 1
    L[1,1], L[2,2] = np.cos(np.deg2rad(2*eta)), np.cos(np.deg2rad(2*eta))
    L[2,1], L[1,2] = np.sin(np.deg2rad(2*eta)), -np.sin(np.deg2rad(2*eta))
    return L


def transformation_angle(theta_in, theta_out, phi_in, phi_out, theta_sca):
    """Calculates the transformation angles for the stokes_rotation_matrix.
    Based on Mishchenko eqn 4.18 and 4.19.
    """

    theta_sca = calc_scattering_angle(theta_in, theta_out, phi_in, phi_out)

    sigma1 = np.argcos((np.cos(np.deg2rad(theta_out)) - np.cos(np.deg2rad(
            theta_in)) * np.cos(np.deg2rad(theta_sca))) / \
            np.sin(np.deg2rad(theta_in)) * np.sin(np.deg2rad(theta_sca)))
    sigma2 = np.argcos((np.cos(np.deg2rad(theta_in)) - np.cos(np.deg2rad(
            theta_out)) * np.cos(np.deg2rad(theta_sca))) / \
            np.sin(np.deg2rad(theta_out)) * np.sin(np.deg2rad(theta_sca)))
    return sigma1, sigma2


def transformed_rayleigh_scattering_matrix(
        theta_in, theta_out, phi_in, phi_out, stokes_dim = 4):
    """ The scattering matrix for the transport coordinate system.
    Based on Mishchenko eqn 4.14.
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

    theta_sca = calc_scattering_angle(theta_in, theta_out, phi_in, phi_out)
    sigma1, sigma2 = transformation_angle(theta_in, theta_out, phi_in,
                                            phi_out, theta_sca)

    L1 = stokes_rotation_matrix(-sigma2)
    L2 = stokes_rotation_matrix(180 - sigma1)
    F = rayleigh_phasematrix(theta_sca)

    P = L1 @ F @ L2
    return P[:stokes_dim,:stokes_dim]


def calc_rayleigh_scattering_cross_section(wavelength):
    """ Calculates the scattering cross section for rayleigh scattering.
    A numerical approximation (accurate to 0.3%) for the Rayleigh scattering
    cross section for air from Stamnes Chapter 3.3.7, the eqn after 3.21.
    The approximation is valid for 0.205 < lambda < 1.05 micrometers.
    (This formula was provided by M. Callan, University of Colorado, who fitted
    the numerical results of Bates (1984).)
    """
    if type(wavelength) not in [int, float]:
        raise TypeError('The wavelengh must be an integer or a float')
    if wavelength < 0:
        raise ValueError('The wavelengh cannot be negative')
    wavelength = wavelength * 1e6
    constants = [3.9729066, 4.6547659e-2, 4.5055995e-4, 2.3229848e-5]
    sum = 0
    for i in range(4):
        sum += constants[i] * wavelength**(-2 * i)
    sigma_ray = wavelength**(-4) * sum * 1e-28
    return sigma_ray * 1e-4 # converted into m^2


def argclosest(value, array, return_value = False):
    """Returns the index in ``array`` which is closest to ``value``."""
    idx = np.abs(array - value).argmin()
    return (idx, array[idx].item()) if return_value else idx


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """ rel_tol is a relative tolerance, it is multiplied by the greater of the
    magnitudes of the two arguments; as the values get larger, so does the
    allowed difference between them while still considering them equal.
    abs_tol is an absolute tolerance that is applied as-is in all cases. If the
    difference is less than either of those tolerances, the values are
    considered equal."""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def delta_func(n):
    """ Delta function (numerical rounding and precision issues are considered
    at the comparioson for float equality)"""
    if type(n) not in [int, float]:
        raise TypeError('Only numbers can be an input!')

    if isclose(n, 0):
        return 1
    else:
        return 0


def plank_freq(freq, temp):
    """ Returns the intensity of electromagnetic radiation at the frequency
    (freq) emitted by a black body in thermal equilibrium at a given
    temperature (temp).
    """
    c_0 = 299_792_458       # m/s
    h = 6.6262*10**(-34)    # J s
    kb = 1.3805*10**(-23)   # J K^-1

    factor = np.exp((h * freq) / (kb * temp))
    B = 2 * h * freq**3 / c_0**2  *  1 / (factor - 1)

    return B


def plank_wavelength(lam, temp):
    """ Returns the intensity of electromagnetic radiation at the wavelength
    (lam) emitted by a black body in thermal equilibrium at a given
    temperature (temp).
    """
    c_0 = 299_792_458       # m/s
    h = 6.6262*10**(-34)    # J s
    kb = 1.3805*10**(-23)   # J K^-1

    factor = np.exp((h * c_0) / (lam * kb * temp))
    B = 2 * h * c_0**2 / lam**5  *  1 / (factor - 1)

    return B


"""
## Converting Units
"""
def km2m(km):
    """Converts km to m"""
    return km * 1000


def m2km(m):
    """Converts m to km"""
    return m / 1000


if __name__ == '__main__':
    # print(rayleigh_phasematrix(90, stokes_dim=2))
    print(calc_rayleigh_scattering_cross_section(500e-9))
    pass

"""
1D Radiative transfer model for shortwave radiation.
"""
import numpy as np
from fRT import constants
from scipy.linalg import expm
from fRT import functions as f

__all__ = ["RT_model_1D"]


class RT_model_1D(object):
    """docstring for RT_model_1D."""

    c_0 = constants.speed_of_light
    h = constants.planck
    kb = constants.boltzmann
    ABSORPTION_CROSS_SEC = 1e-30
    SCATTERING_CROSS_SEC_HENYEY = 6.24e-32

    def __init__(self):
        super().__init__()
        # Radiation specific
        self.wavelength = None
        self.scattering_cross_sec = None

        # init over different functions
        # Ground informations
        self.ground_albedo = 0.7
        self.reflection_type = 0  # lambert

        # receiver
        self.receiver_height = None
        self.receiver_elevation = None
        self.receiver_azimuth = None

        # sun
        self.sun_intensity = None
        self.sun_elevation = 0
        self.sun_azimuth = 90

        # atm fields
        self.swiping_height = None
        self.height_array = None
        self.absorption_coeff_field = None
        self.scattering_coeff_field = None
        self.temp_field = None

        # model controll
        self.use_planck = 1
        self.use_scat = 1
        self.scat_type = 1
        self.stokes_dim = 1

    def set_stokes_dim(self, stokes_dim):
        """Sets the dimention of the stokes vector."""
        if stokes_dim not in [1, 2, 3, 4]:
            raise ValueError(
                "The dimention of the stokes vector can only be 1, 2, 3 or 4"
            )

        self.stokes_dim = stokes_dim

    def set_scattering_type(self, scattering_type):
        """Set the scattering type for the model
        Options are scattering with an constant scattering cross section and
        a phasefunction based on Henyey Greenstein (0) and rayleigh
        scattering with an wavelength dependent cross section (1).
        For full polarimetric simulation (stokes_dim > 1) rayleigh
        scattering will be used.
        """
        if scattering_type == "rayleigh":
            scattering_type = 1
        elif scattering_type == "henyey_greenstein":
            scattering_type = 0
        if scattering_type not in [0, 1]:
            raise ValueError(
                'Only "henyey_greenstein" (0) or "rayleigh" (1) \
                            are valid options'
            )
        if self.stokes_dim in [
            2,
            3,
            4,
        ]:  # full polarimetric can only be rayleigh scattering
            scattering_type = 1

        self.scat_type = scattering_type

    def toggle_planck_radiation(self, setting=1):
        """Toggle the use of planck radiation in the model.
        The default is on (1).
        """
        if setting not in [0, 1, "on", "off"]:
            raise ValueError(
                "The setting for the toggle the use of planck "
                'Radiation must "on" (1) or "off" (0)'
            )
        self.use_planck = 1 if setting == "on" else 0 if setting == "off" else setting

    def toggle_scattering(self, setting=1):
        """Toggle the use of scattering in the model. (So k = abs_coef)
        The default is on (1).
        """
        if setting not in [0, 1, "on", "off"]:
            raise ValueError(
                "The input for the toggle the us of scattering "
                'in the model must "on" (1) or "off" (0)'
            )
        self.use_scat = 1 if setting == "on" else 0 if setting == "off" else setting

    def set_wavelenth(self, wavelength):
        """Sets the wavelength for the model instance.

        This is needed for the scattering cross section as well as the incoming sun
        intensity.

        Parameters:
            wavelength (float) [m]:
                The wavelength for which the simulation will be performed.
        """
        if wavelength < 0:
            raise ValueError("The wavelength cannot be negative")

        self.wavelength = wavelength
        RT_model_1D.set_scattering_cross_sec(self)
        RT_model_1D.get_atmoshperic_profiles(self)
        self.sun_intensity = f.sun_init_intensity(self.wavelength, self.stokes_dim)

    def set_scattering_cross_sec(self):
        """Sets the scattering cross section according to the wavelength of the
        model instance."""
        if self.scat_type == 1:
            sigma = f.calc_rayleigh_scattering_cross_section(self.wavelength)
            self.scattering_cross_sec = sigma
        elif self.scat_type == 0:
            self.scattering_cross_sec = self.SCATTERING_CROSS_SEC_HENYEY

    def set_reflection_type(self, reflection_type=0):
        """Set the reflection type for the model
        Options are lambert (0) and specular (1) and a linear combination of
        those two.
        """
        if reflection_type < 0 or reflection_type > 1:
            raise ValueError("Must be 0, 1 or inbetween")

        self.reflection_type = reflection_type

    def set_receiver(self, height, elevation, azimuth):
        """Sets the position (closest to the model grid) and viewing angle of
        the receiver e.g. where and in which direction the radiation field is
        evaluated. The function converts the angle into the radiation transport
        direction.

        Parameters:
            height (float): Height of the receiver (must be positive) [km].
            elevation (float):  The elevation angle the receiver is looking at.
                            Must be between 0 (up) and 180 (down) [deg].
            azimuth (float):  The azimuth angle the receiver is looking at.
                            Must be between 0 and 360 [deg].
        """

        if height < 0:
            raise ValueError("The height cannot be negative")
        if elevation < 0 or elevation > 180:
            raise ValueError("The elevation cannot be negative or greater 180")
        if elevation == 90:
            raise ValueError("The elevation can not be 90")
        if azimuth < 0 or azimuth >= 360:
            raise ValueError("The azimuth cannot be negative or >= 360")

        theta, phi = f.convert_direction(elevation, azimuth)

        idx, height = f.argclosest(f.km2m(height), self.height_array, return_value=True)
        self.receiver_height = height
        self.receiver_elevation = theta
        self.receiver_azimuth = phi

    def set_sun_position(self, elevation, azimuth, intensity=None):
        """Sets the elevation and azimuth angle of the sun and the intensity.
        The function converts the angle into the radiation transport direction.

        Parameters:
            elevation (float):  The elevation angle of the sun.
                            Must be between 0 (zenith) and 90 (horizion) [deg].
            azimuth (float):  The azimuth angle of the sun.
                            Must be between 0 and 360 [deg].
            intensity (float): Intensity of the sun (positive) [W/m2/sr/nm].
        """

        if elevation < 0 or elevation >= 90:
            raise ValueError("The elevation cannot be negative or >= 90")
        if azimuth < 0 or azimuth >= 360:
            raise ValueError("The azimuth cannot be negative or >= 360")
        if intensity is not None:
            if intensity < 0:
                raise ValueError("The intensity cannot be negative")

        theta, phi = f.convert_direction(elevation, azimuth)

        self.sun_elevation = theta
        self.sun_azimuth = phi
        if intensity is not None and self.sun_intensity is not None:
            print(
                "The set sun intensity might not fit to the suns intensity \
                    at the set wavelengh"
            )
            self.sun_intensity = np.zeros((self.stokes_dim))
            self.sun_intensity[0] = intensity
        elif intensity is not None and self.sun_intensity is None:
            self.sun_intensity = np.zeros((self.stokes_dim))
            self.sun_intensity[0] = intensity
        else:
            self.sun_intensity = f.sun_init_intensity(self.wavelength, self.stokes_dim)

    def define_grid(self, atm_height=200, swiping_height=1):
        """Sets the Grid for the atmosheric parameters.

        Parameters:
            atm_height (float): Total height [km].
            swiping_height (float): The height of an atm layer in witch the atm
                                parameters stay constant [km].
        Returns:
            ndarray:
                height array [m]
        """

        if atm_height < 0:
            raise ValueError("The height cannot be negative")
        if swiping_height < 0:
            raise ValueError("The swiping height must be positive")

        atm_height = f.km2m(atm_height)
        self.swiping_height = f.km2m(swiping_height)
        self.height_array = np.arange(
            0, atm_height + self.swiping_height, self.swiping_height
        )

        return self.height_array

    def get_atmoshperic_profiles(self):
        """Returns atm fields of the absorption and scattering coefficent
        depending on the readin_densprofile"""
        self.absorption_coeff_field = np.zeros(
            (len(self.height_array), self.stokes_dim, self.stokes_dim)
        )
        self.scattering_coeff_field = np.zeros(
            (len(self.height_array), self.stokes_dim, self.stokes_dim)
        )
        dens_profile_height, dens_profile = f.readin_densprofile()

        for idx, height in enumerate(self.height_array):
            dens = dens_profile[f.argclosest(height, f.km2m(dens_profile_height))]
            np.fill_diagonal(
                self.absorption_coeff_field[idx], dens * self.ABSORPTION_CROSS_SEC
            )
            np.fill_diagonal(
                self.scattering_coeff_field[idx], dens * self.scattering_cross_sec
            )

        return self.absorption_coeff_field, self.scattering_coeff_field

    def set_atmosheric_temp_profile(self):
        """DocString"""
        self.temp_field = np.empty((len(self.height_array)))
        temp_profile_height, temp_profile = f.readin_tempprofile()

        for idx, height in enumerate(self.height_array):
            self.temp_field[idx] = temp_profile[
                f.argclosest(height, f.km2m(temp_profile_height))
            ]

        return self.temp_field

    def calc_direct_beam_intensity(self, height):
        """Calculates the intensity of the suns beam at the given height with
        just extinction. This is needed for the Scattering source term.

        Parameters:
            height (float): Height where the direct beam is evaluated [m].

        Returns:
            I_dir (float): The intensity of the direct beam at that height.
        """
        if height < 0:
            raise ValueError("The height cannot be negative")

        idx, height = f.argclosest(height, self.height_array, return_value=True)
        angle, _ = f.convert_direction(self.sun_elevation, self.sun_azimuth)

        tau = np.zeros((self.stokes_dim, self.stokes_dim))
        for lvl in np.arange(len(self.height_array) - 1, idx - 1, -1):
            tau += (
                (
                    self.absorption_coeff_field[lvl]
                    + self.scattering_coeff_field[lvl] * self.use_scat
                )
                * self.swiping_height
                / np.cos(np.deg2rad(angle))
            )

        if self.stokes_dim == 1:
            I_dir = self.sun_intensity * np.exp(-tau)
        else:
            I_dir = expm(-tau) @ self.sun_intensity
        return I_dir

    def scattering_source_term(self, height):
        idx, height = f.argclosest(height, self.height_array, return_value=True)

        angle = f.calc_scattering_angle(
            self.sun_elevation,
            self.receiver_elevation,
            self.sun_azimuth,
            self.receiver_azimuth,
        )

        # setting the phase function according to the selected scattering type
        if self.scat_type:
            phase_func = f.transformed_rayleigh_scattering_matrix(
                self.sun_elevation,
                self.receiver_elevation,
                self.sun_azimuth,
                self.receiver_azimuth,
                self.stokes_dim,
            )
            # phase_func = f.rayleigh_phasematrix(angle, self.stokes_dim)
        else:
            phase_func = f.henyey_greenstein_phasefunc(angle, g=0.7)

        if self.stokes_dim == 1:
            I_scat = (
                (1 - np.exp(-self.scattering_coeff_field[idx] * self.swiping_height))
                * RT_model_1D.calc_direct_beam_intensity(self, height)
                * phase_func
            )
        else:
            I_scat = (
                (np.eye(self.stokes_dim) - expm(-self.scattering_coeff_field[idx] *
                                                self.swiping_height))
                @ RT_model_1D.calc_direct_beam_intensity(self, height)
                @ phase_func
            )

        return I_scat

    def planck_source_term(self, height):
        idx, height = f.argclosest(height, self.height_array, return_value=True)
        temp = self.temp_field[idx]

        return f.planck_wavelength(self.wavelength, temp, self.stokes_dim)

    def extinction_term(self, intensity, height):
        """Clalculates the extinction term based on the given intensity and the
        absorbtion and scattering coefficent at the given height."""

        id = f.argclosest(height, self.height_array)
        k = (
            self.absorption_coeff_field[id]
            + self.use_scat * self.scattering_coeff_field[id]
        )
        # np.exp calculates element wise not the matricexponential!
        # but for diagonal matricies this is the same
        if self.stokes_dim == 1:
            I_ext = intensity * np.exp(-k * self.swiping_height)
        else:
            I_ext = expm(-k * self.swiping_height) @ intensity

        return I_ext

    def create_receiver_viewing_field(self):
        """creates an empty array where the field will be evaluated"""
        height = self.receiver_height
        # revert in viewing direct
        angle, _ = f.convert_direction(self.receiver_elevation, self.receiver_azimuth)

        if angle < 90:
            # from rec (at idx) to TOA (len(h.a.))
            height_at_rad_field = np.arange(
                self.height_array[-1],
                height - self.swiping_height,
                -self.swiping_height,
            )

        elif angle > 90:
            # from ground (at 0) to rec (at idx)
            height_at_rad_field = np.arange(
                0, height + self.swiping_height, self.swiping_height
            )

        else:
            height_at_rad_field = np.NAN

        return height_at_rad_field

    def get_receiver_viewing_field(self):
        """ Returns the height field for the it is seen from the receiver """
        field = RT_model_1D.create_receiver_viewing_field(self)
        return np.flipud(field)

    def rad_field_initial_condition(self):
        """Returns the starting value based on where the reciever is looking"""

        # revert in viewing direct
        angle, _ = f.convert_direction(self.receiver_elevation, self.receiver_azimuth)
        # Looking at the sky
        if angle < 90:
            I_init = (
                self.sun_intensity
                * f.delta_func(self.sun_elevation - self.receiver_elevation)
                * f.delta_func(self.sun_azimuth - self.receiver_azimuth)
            )

        # Looking at the ground
        elif angle > 90:
            I_ground = RT_model_1D.calc_direct_beam_intensity(self, 0)

            I_lambert = (
                I_ground
                * self.ground_albedo
                * np.cos(np.deg2rad((self.sun_elevation + 180) % 360))
            )

            I_specular = (
                I_ground
                * self.ground_albedo
                * f.delta_func(self.sun_elevation + self.receiver_elevation - 180)
                * f.delta_func(self.sun_azimuth - self.receiver_azimuth)
            )

            I_init = (
                1 - self.reflection_type
            ) * I_lambert + self.reflection_type * I_specular

        else:
            I_init = np.empty(self.stokes_dim)
            I_init.fill(np.nan)

        return I_init

    def evaluate_radiation_field(self):
        """DocString"""
        height_at_rad_field = RT_model_1D.create_receiver_viewing_field(self)
        rad_field = np.empty((len(height_at_rad_field), self.stokes_dim))

        rad_field[0] = RT_model_1D.rad_field_initial_condition(self)

        for id, height in enumerate(height_at_rad_field[1:]):
            ext = RT_model_1D.extinction_term(self, rad_field[id], height)
            scat = RT_model_1D.scattering_source_term(self, height)
            planck = RT_model_1D.planck_source_term(self, height)
            # id starts at 0 for idx 1 from height at rad field!
            rad_field[id + 1] = ext + self.use_scat * scat + self.use_planck * planck

        # invert the rad_field for the uplooking case
        return np.flipud(rad_field)

    def print_testvals(self):
        """prints an attribute which must be set here. It's for testing."""
        print(self.use_planck)

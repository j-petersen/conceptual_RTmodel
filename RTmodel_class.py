'''
1D Radiative transfer model for shortwave radiation.
'''
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt
from submodules import functions as f
from submodules import plotting_routines as pr

def main():
    # Control parameters
    sun_ele, sun_azi = 45, 180   # position in sky [degree, 0° = zenith]
    sun_intensity = 1000       # intensity of the sun [W/m2sr]

    receiver_height = 0   # height of the receiver [km]
    receiver_elevation_angle = 45  # viewing angle [degree, 180° = upward rad]
    receiver_azimuth_angle = 180


    model = RT_model_1D()   # create model instance

    lvl_grid = model.define_grid(atm_height = 200, swiping_height = 1)

    model.set_wavelenth(500*10**(-9))

    model.get_atmoshperic_profiles()
    model.set_atmosheric_temp_profile()

    model.toggle_plank_radiation('on')
    model.toggle_scattering('on')

    model.set_reflection_type(0)

    model.set_sun_position(sun_intensity, sun_ele, sun_azi)
    model.set_receiver(receiver_height, receiver_elevation_angle,
                        receiver_azimuth_angle)


    rad_field = model.evaluate_radiation_field()
    # print(rad_field)

    model.print_testvals()


class RT_model_1D(object):
    """docstring for RT_model_1D."""
    c_0 = 299_792_458       # m/s
    h = 6.6262*10**(-34)    # J s
    kb = 1.3805*10**(-23)   # J K^-1
    ABSORPTION_CROSS_SEC = 1e-30
    SCATTERIN_CROSS_SEC_HENYEY = 6.24e-32

    def __init__(self):
        super().__init__()
        ## Radiation specific
        self.wavelength = None
        self.scattering_cross_sec = None


        ## init over different functions
        # Ground informations
        self.ground_albedo = 0.7
        self.reflection_type = 0        # lambert

        # receiver
        self.receiver_height = None
        self.receiver_elevation = None
        self.receiver_azimuth = None

        # sun
        self.sun_intensity = 1000
        self.sun_elevation = 0
        self.sun_azimuth = 90

        # atm fields
        self.swiping_height = None
        self.height_array = None
        self.absorption_coeff_field = None
        self.scattering_coeff_field = None
        self.temp_field = None

        ## model controll
        self.use_plank = 1
        self.use_scat = 1
        self.scat_type = 1


    def set_scattering_type(self, scattering_type):
        """ Set the scattering type for the model
            Options are scattering with an constant scattering cross section and
            a phasefunction based on Henyey Greenstein (0) and rayleigh
            scattering with an wavelength dependent cross section (1).
            For full polarimetric simulation (stokes_dim > 1) rayleigh
            scattering will be used.
        """
        if scattering_type == 'rayleigh':
            scattering_type = 1
        elif scattering_type == 'henyey_greenstein':
            scattering_type = 0
        if scattering_type not in [0,1]:
            raise ValueError('Only "henyey_greenstein" (0) or "rayleigh" (1) \
                            are valid options')

        self.scat_type = scattering_type


    def toggle_plank_radiation(self, input):
        """ Toggle the use of plank radiation in the model.
        The default is on (1).
        """
        if input not in [0, 1, 'on', 'off']:
            raise ValueError('The input for the toggle the use of plank '\
                    'Radiation must "on" (1) or "off" (0)')
        self.use_plank = 1 if input == 'on' else 0 if input == 'off' else input


    def toggle_scattering(self, input):
        """ Toggle the use of scattering in the model. (So k = abs_coef)
        The default is on (1).
        """
        if input not in [0, 1, 'on', 'off']:
            raise ValueError('The input for the toggle the us of scattering '\
                    'in the model must "on" (1) or "off" (0)')
        self.use_scat = 1 if input == 'on' else 0 if input == 'off' else input


    def set_wavelenth(self, wavelength):
        """ Sets the wavelength for the model instance.

        Parameters:
            wavelength (float) [m]:
                The wavelength for which the simulation will be performed.
        """
        if type(wavelength) not in [int, float]:
            raise TypeError('The wavelength must be an int or float')
        if wavelength < 0:
            raise ValueError('The wavelength cannot be negative')

        self.wavelength = wavelength
        RT_model_1D.set_scattering_cross_sec(self)
        RT_model_1D.get_atmoshperic_profiles(self)


    def set_scattering_cross_sec(self):
        """ Sets the scattering cross section according to the wavelength of the
        model instance."""
        if self.scat_type == 1:
            sigma = f.calc_rayleigh_scattering_cross_section(self.wavelength)
            self.scattering_cross_sec = sigma
        elif self.scat_type == 0:
            self.scattering_cross_sec = self.SCATTERIN_CROSS_SEC_HENYEY



    def set_reflection_type(self, reflection_type = 0):
        """Set the reflection type for the model
            Options are lambert (0) and specular (1) and a linear combination of
            those two.
        """
        if type(reflection_type) not in [int, float]:
            raise TypeError('Must be an int or an float')
        if reflection_type < 0 or reflection_type >1:
            raise ValueError('Must be 0, 1 or inbetween')

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

        if type(height) not in [int, float]:
            raise TypeError('The height must be a non-negative real number')
        if height < 0:
            raise ValueError('The height cannot be negative')
        if type(elevation) not in [int, float]:
            raise TypeError('An angle must be a real number between 0 and 180')
        if elevation < 0 or elevation > 180:
            raise ValueError('The elevation cannot be negative or greater 180')
        if elevation == 90:
            raise ValueError('The elevation can not be 90')
        if type(azimuth) not in [int, float]:
            raise TypeError('An angle must be a real number between 0 and 360')
        if azimuth < 0 or azimuth >= 360:
            raise ValueError('The azimuth cannot be negative or >= 360')

        idx, height = f.argclosest(f.km2m(height), self.height_array,
                                    return_value = True)
        self.receiver_height = height
        self.receiver_elevation = (elevation + 90) % 180
        self.receiver_azimuth = (azimuth + 180) % 360


    def set_sun_position(self, intensity, elevation, azimuth):
        """Sets the elevation and azimuth angle of the sun and the intensity.
        The function converts the angle into the radiation transport direction.

        Parameters:
            intensity (float): Intensity of the sun (positive) [W/m2/sr/nm].
            elevation (float):  The elevation angle of the sun.
                            Must be between 0 (zenith) and 90 (horizion) [deg].
            azimuth (float):  The azimuth angle of the sun.
                            Must be between 0 and 360 [deg].
        """

        if type(intensity) not in [int, float]:
            raise TypeError('The intensity must be a non-negative real number')
        if intensity < 0:
            raise ValueError('The intensity cannot be negative')
        if type(elevation) not in [int, float]:
            raise TypeError('The elevation angle must an int or float')
        if elevation < 0 or elevation >= 90:
            raise ValueError('The elevation cannot be negative or >= 90')
        if type(azimuth) not in [int, float]:
            raise TypeError('The azimuth angle must an int or float')
        if azimuth < 0 or azimuth >= 360:
            raise ValueError('The azimuth cannot be negative or >= 360')

        self.sun_intensity = intensity
        self.sun_elevation = (elevation + 90) % 180
        self.sun_azimuth = (azimuth + 180) % 360


    def define_grid(self, atm_height = 200, swiping_height = 1):
        """Sets the Grid for the atmosheric parameters.

        Parameters:
            atm_height (float): Total height [km].
            swiping_height (float): The height of an atm layer in witch the atm
                                parameters stay constant [km].
        Returns:
            ndarray:
                height array [m]
        """

        if type(atm_height) not in [int, float]:
            raise TypeError('The height must be a non-negative real number')
        if atm_height < 0:
            raise ValueError('The height cannot be negative')
        if type(swiping_height) not in [int, float]:
            raise TypeError('The swiping height must be a positive real number')
        if swiping_height < 0:
            raise ValueError('The swiping height must be positive')

        atm_height = f.km2m(atm_height)
        self.swiping_height = f.km2m(swiping_height)
        self.height_array = np.arange(0, atm_height + self.swiping_height,
                                    self.swiping_height)

        return self.height_array


    def get_atmoshperic_profiles(self):
        """ Returns atm fields of the absorption and scattering coefficent
        depending on the readin_densprofile """
        self.absorption_coeff_field = np.empty((len(self.height_array)))
        self.scattering_coeff_field = np.empty((len(self.height_array)))
        dens_profile_height, dens_profile = f.readin_densprofile()

        for idx, height in enumerate(self.height_array):
            dens = dens_profile[f.argclosest(height,
                                f.km2m(dens_profile_height))]
            print(dens)
            self.absorption_coeff_field[idx] = dens * self.ABSORPTION_CROSS_SEC
            self.scattering_coeff_field[idx] = dens * self.scattering_cross_sec

        return self.absorption_coeff_field, self.scattering_coeff_field


    def set_atmosheric_temp_profile(self):
        """DocString"""
        self.temp_field = np.empty((len(self.height_array)))
        temp_profile_height, temp_profile = f.readin_tempprofile()

        for idx, height in enumerate(self.height_array):
            self.temp_field[idx] = temp_profile[f.argclosest(height,
                                                f.km2m(temp_profile_height))]

        return self.temp_field


    def calc_direct_beam_intensity(self, height):
        """ Calculates the intensity of the suns beam at the given height with
        just extinction. This is needed for the Scattering source term.

        Parameters:
            height (float): Height where the direct beam is evaluated [m].

        Returns:
            I_dir (float): The intensity of the direct beam at that height.
        """
        if type(height) not in [int, float]:
            raise TypeError('The height must be a non-negative real number')
        if height < 0:
            raise ValueError('The height cannot be negative')

        idx, height = f.argclosest(height, self.height_array,
                                    return_value = True)
        angle = (self.sun_elevation + 90) % 180

        tau = 0
        for lvl in np.arange(len(self.height_array)-1,idx-1,-1):

            tau += (self.absorption_coeff_field[lvl] + \
                self.scattering_coeff_field[lvl] * self.use_scat) * \
                self.swiping_height / np.cos(np.deg2rad(angle))

        I_dir = self.sun_intensity * np.exp(- tau)

        return I_dir


    def scattering_source_term(self, height):
        idx, height = f.argclosest(height, self.height_array,
                                    return_value = True)

        angle = f.calc_scattering_angle(self.sun_elevation,
            self.receiver_elevation,self.sun_azimuth, self.receiver_azimuth)

        # setting the phase function according to the selected scattering type
        if self.scat_type:
            phase_func = f.rayleigh_phasematrix(angle, stokes_dim = 1)
        else:
            phase_func = f.henyey_greenstein_phasefunc(angle, g = 0.7)

        I_scat = (1 - np.exp(-self.scattering_coeff_field[idx] *
                             self.swiping_height)) * \
                    RT_model_1D.calc_direct_beam_intensity(self, height) *\
                    phase_func

        return I_scat


    def plank_source_term(self, height):
        idx, height = f.argclosest(height, self.height_array,
                                    return_value = True)
        temp = self.temp_field[idx]

        return f.plank_wavelength(self.wavelength, temp)


    def extinction_term(self, intensity, height):
        """Clalculates the extinction term based on the given intensity and the
        absorbtion and scattering coefficent at the given height. """

        id = f.argclosest(height, self.height_array)
        k = self.absorption_coeff_field[id] + \
            self.use_scat * self.scattering_coeff_field[id]

        I_ext = intensity * np.exp(-k * self.swiping_height)

        return I_ext


    def create_receiver_viewing_field(self):
        """creates an empty array where the field will be evaluated"""
        height = self.receiver_height
        angle = (self.receiver_elevation + 90) % 180 # revert in viewing direct
        idx = f.argclosest(self.receiver_height, self.height_array)

        if angle < 90:
            # from rec (at idx) to TOA (len(h.a.))
            height_at_rad_field = np.arange(self.height_array[-1], height - \
                                    self.swiping_height, -self.swiping_height)

        elif angle > 90:
            # from ground (at 0) to rec (at idx)
            height_at_rad_field = np.arange(0, height + self.swiping_height,
                                        self.swiping_height)


        return  height_at_rad_field



    def get_receiver_viewing_field(self):
        """ Returns the height field for the it is seen from the receiver """
        field = RT_model_1D.create_receiver_viewing_field(self)
        return np.flipud(field)



    def rad_field_initial_condition(self):
        """Returns the starting value based on where the reciever is looking"""

        angle = (self.receiver_elevation + 90) % 180 # revert in viewing direct
        # Looking at the sky
        if angle < 90:
            I_init = self.sun_intensity * f.delta_func(
                self.sun_elevation - self.receiver_elevation) * \
                f.delta_func(self.sun_azimuth - self.receiver_azimuth)

        # Looking at the ground
        elif angle > 90:
            I_ground = RT_model_1D.calc_direct_beam_intensity(self, 0)

            I_lambert = I_ground * self.ground_albedo * \
                np.cos(np.deg2rad((self.sun_elevation + 180) % 360))

            I_specular = I_ground * self.ground_albedo * \
                f.delta_func(self.sun_elevation +
                self.receiver_elevation - 180) * \
                f.delta_func(self.sun_azimuth - self.receiver_azimuth)

            I_init = (1 - self.reflection_type) * I_lambert + \
                     self.reflection_type * I_specular


        return I_init


    def evaluate_radiation_field(self):
        """DocString"""
        angle = (self.receiver_elevation + 90) % 180 # revert in viewing direct

        height_at_rad_field = RT_model_1D.create_receiver_viewing_field(self)
        rad_field = np.empty((len(height_at_rad_field)))

        rad_field[0] = RT_model_1D.rad_field_initial_condition(self)

        for id, height in enumerate(height_at_rad_field[1:]):
            # id starts at 0 for idx 1 from height at rad field!
            rad_field[id+1] = RT_model_1D.extinction_term(self, rad_field[id],
                    height) + self.use_scat * \
                    RT_model_1D.scattering_source_term(self, height) + \
                    self.use_plank * RT_model_1D.plank_source_term(self, height)

        # invert the rad_field for the uplooking case
        return np.flipud(rad_field)


    def print_testvals(self):
        """prints an attribute which must be set here. It's for testing."""
        print(self.use_plank)


if __name__ == '__main__':
    main()

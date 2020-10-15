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
    sun_ele, sun_azi = 45, 0   # position in sky [degree, 0° = zenith]
    sun_intensity = 1000       # intensity of the sun [W/m2sr]

    receiver_height = 0   # height of the receiver [km]
    receiver_elevation_angle = 135  # viewing angle [degree, 180° = upward rad]
    receiver_azimuth_angle = 180


    model = RT_model_1D()   # create model instance

    lvl_grid = model.define_grid(atm_height = 200, swiping_height = 1)

    model.get_atmoshpere_profils()

    model.set_reflection_type(0)

    model.set_wavelenth(500*10**(-9))

    model.set_sun_position(sun_intensity, sun_ele, sun_azi)
    model.set_receiver(receiver_height, receiver_elevation_angle,
                        receiver_azimuth_angle)


    rad_field = model.evaluate_radiation_field()
    #model.print_testvals()
    print(rad_field)

    int_grid = np.empty((len(lvl_grid)))
    source_grid = np.empty((len(lvl_grid)))

    for id, lvl in enumerate(lvl_grid):
        int_grid[id] = model.calc_direct_beam_intensity(int(lvl))
        source_grid[id] = model.scattering_source_term(lvl)

    #print(int_grid)

    if False:
        ty.plots.styles.use(["typhon", 'typhon-dark'])
        fig, ax = plt.subplots(ncols=1,nrows=1)
        pr.plot_intensity(list(int_grid), list(lvl_grid/1000),
                        ax = ax, multi_way = False)
        #fig.savefig()
        plt.show()



class RT_model_1D(object):
    """docstring for RT_model_1D."""

    def __init__(self):
        super().__init__()
        ## Radiation specific
        self.wavelength = None
        self.absorption_cross_sec = 1e-30
        self.scattering_cross_sec = 6.24e-32


        ## init over different functions
        # Parameter
        self.ground_albedo = 0.7
        self.reflection_type = 0        # lambert

        # receiver
        self.receiver_height = None
        self.receiver_elevation = None
        self.receiver_azimuth = None

        # sun
        self.sun_intensity = 1000
        self.sun_elevation = 0          # downward
        self.sun_azimuth = 90

        # atm fields
        self.swiping_height = None
        self.height_array = None
        self.absorption_coeff_field = None
        self.scattering_coeff_field = None
        self.temp_field = None


    def set_wavelenth(self, wavelength):
        """Sets the wavelength for the model instance.

        Parameters:
            wavelength (float) [m]:
                The wavelength for which the simulation will be performed.
        """
        if type(wavelength) not in [int, float]:
            raise TypeError('The wavelength must be an int or float')
        if wavelength < 0:
            raise ValueError('The wavelength cannot be negative')

        self.wavelength = wavelength

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
            raise TypeError('The angle must be a real number between 0 and 180')
        if elevation < 0 or elevation > 180:
            raise ValueError('The angle cannot be negative or greater 180')
        if type(azimuth) not in [int, float]:
            raise TypeError('The angle must be a real number between 0 and 360')
        if azimuth < 0 or azimuth >= 360:
            raise ValueError('The angle cannot be negative or >= 360')

        idx, height = RT_model_1D.argclosest(RT_model_1D.km2m(height),
                                self.height_array, return_value = True)
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
        self.sun_elevation = (elevation + 90) % 90
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

        atm_height = RT_model_1D.km2m(atm_height)
        self.swiping_height = RT_model_1D.km2m(swiping_height)
        self.height_array = np.arange(0, atm_height + self.swiping_height,
                                    self.swiping_height)

        return self.height_array


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

        idx, height = RT_model_1D.argclosest(height,
                                self.height_array, return_value = True)

        tau = 0
        for lvl in np.arange(len(self.height_array)-1,idx-1,-1):

            tau += (self.absorption_coeff_field[lvl] + \
                self.scattering_coeff_field[lvl]) * \
                self.swiping_height / \
                np.cos(RT_model_1D.deg2rad(self.sun_elevation)) # revert to viewing direct

        I_dir = self.sun_intensity * np.exp(- tau)

        return I_dir


    def scattering_source_term(self, height):
        idx, height = RT_model_1D.argclosest(height,
                                self.height_array, return_value = True)

        angle = RT_model_1D.calc_scattering_angle(self.sun_elevation,
                                                self.receiver_elevation,
                                                self.sun_azimuth,
                                                self.receiver_azimuth)

        I_scat = (1 - np.exp(-self.scattering_coeff_field[idx] *
                             self.swiping_height)) * \
                    RT_model_1D.calc_direct_beam_intensity(self, height) *\
                    RT_model_1D.phasefunc(angle)
        return I_scat


    def extinction_term(self, intensity, height):
        """Clalculates the extinction term based on the given intensity and the
        absorbtion and scattering coefficent at the given height. """

        id = RT_model_1D.argclosest(height, self.height_array)
        k = self.absorption_coeff_field[id] + self.scattering_coeff_field[id]

        I_ext = intensity * np.exp(-k * self.swiping_height)

        return I_ext


    def get_atmoshpere_profils(self):
        """ Returns atm fields of the absorption and scattering coefficent
        depending on the readin_densprofile """
        self.absorption_coeff_field = np.empty((len(self.height_array)))
        self.scattering_coeff_field = np.empty((len(self.height_array)))
        dens_profil, dens_profil_height = RT_model_1D.readin_densprofile()

        for idx, height in enumerate(self.height_array):
            dens = dens_profil[RT_model_1D.argclosest(height,
                            RT_model_1D.km2m(dens_profil_height))]
            self.absorption_coeff_field[idx] = dens * self.absorption_cross_sec
            self.scattering_coeff_field[idx] = dens * self.scattering_cross_sec

        return self.absorption_coeff_field, self.scattering_coeff_field


    def create_reciever_viewing_field(self):
        """create an empty array where the field will be evaluated"""
        height = self.receiver_height
        angle = (self.receiver_elevation + 90) % 180 # revert in viewing direct
        idx = RT_model_1D.argclosest(self.receiver_height, self.height_array)

        if angle < 90:
            # from rec (at idx) to TOA (len(h.a.))
            height_at_rad_field = np.arange(self.height_array[-1], height - \
                                    self.swiping_height, -self.swiping_height)

        elif angle > 90:
            # from ground (at 0) to rec (at idx)
            height_at_rad_field = np.arange(0, height + self.swiping_height,
                                        self.swiping_height)

        elif angle == 90:
            # homogenius at one level
            height_at_rad_field = height

        return  height_at_rad_field


    def rad_field_initial_candition(self):
        """Returns the starting value based on where the reciever is looking"""

        angle = (self.receiver_elevation + 90) % 180 # revert in viewing direct
        # Looking at the sky
        if angle < 90:
            I_init = self.sun_intensity * RT_model_1D.delta_func(
                self.sun_elevation - self.receiver_elevation) * \
                RT_model_1D.delta_func(self.sun_azimuth - self.receiver_azimuth)

        # Looking at the ground
        elif angle > 90:
            I_ground = RT_model_1D.calc_direct_beam_intensity(self, 0)
            I_lambert = I_ground * self.ground_albedo * \
                np.cos(RT_model_1D.deg2rad(self.receiver_elevation))
            I_specular = I_ground * self.ground_albedo * \
                RT_model_1D.delta_func((self.sun_elevation -
                self.receiver_elevation + 90) % 180) * \
                RT_model_1D.delta_func(self.sun_azimuth - self.receiver_azimuth)

            I_init = (1 - self.reflection_type) * I_lambert + \
                     self.reflection_type * I_specular

        # Looking along the slap
        elif angle == 90:
            pass

        return I_init


    def evaluate_radiation_field(self):
        """DocString"""
        angle = (self.receiver_elevation + 90) % 180 # revert in viewing direct

        height_at_rad_field = RT_model_1D.create_reciever_viewing_field(self)
        rad_field = np.empty((len(height_at_rad_field)))

        rad_field[0] = RT_model_1D.rad_field_initial_candition(self)

        for id, height in enumerate(height_at_rad_field[1:]):
            # id starts at 0 for idx 1 from height at rad field!
            rad_field[id+1] = RT_model_1D.extinction_term(self, rad_field[id],
                    height) + RT_model_1D.scattering_source_term(self, height)

        # invert the rad_field for the uplooking case
        return np.flipud(rad_field) if angle < 90 else rad_field


    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

        return float( RT_model_1D.rad2deg( np.arccos(angle) ) )

    def print_testvals(self):
        """prints an attribute which must be set here. It's for testing."""
        print(self.sun_azimuth)

    @staticmethod
    def argclosest(value, array, return_value = False):
        """Returns the index in ``array`` which is closest to ``value``."""
        idx = np.abs(array - value).argmin()
        return (idx, array[idx].item()) if return_value else idx

    @staticmethod
    def deg2rad(deg):
        """Converts degress to radian"""
        return deg * np.pi / 180

    @staticmethod
    def rad2deg(rad):
        """Converts degress to radian"""
        return rad * 180 / np.pi

    @staticmethod
    def km2m(km):
        """Converts km to m"""
        return km * 1000

    @staticmethod
    def m2km(m):
        """Converts m to km"""
        return m / 1000

    @staticmethod
    def delta_func(n):
        """ Delta function """
        if type(n) not in [int, float]:
            raise TypeError('Only numbers can be an input!')

        if n == 0:
            return 1
        else:
            return 0

if __name__ == '__main__':
    main()

'''
1D Radiative transfer model for shortwave radiation.
'''
import numpy as np
import matplotlib.pyplot as plt
import typhon as ty
from submodules import plotting_routines as pr
from submodules import functions as f

# look in /data/share/lehre/unix/rtcourse/typhon/typhon/arts/scattering
# https://github.com/atmtools/arts/blob/master/src/m_rte.cc

def main():
    # Control parameters
    sun_angle = 0  # viewing angle [degree, 0° = downward radiation]
    sun_intensity = 1000  # intensity of the sun [W/m2sr]

    model = RT_model_1D()

    lvl_grid = model.define_grid()

    model.get_atmoshpere_profils()

    model.set_sun(sun_intensity, sun_angle)

    model.set_reflection_type('specular')

    #print(model.calc_scattering_angle(30))

    int_grid = np.zeros((len(lvl_grid)))
    for id, lvl in enumerate(RT_model_1D.m2km(lvl_grid)):
        int_grid[id] = model.direkt_beam_intensity(lvl)


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
        ## Constant Parameters
        self.absorption_cross_sec = 1e-30
        self.scattering_cross_sec = 6.24e-32

        ## init over different functions
        # Parameter
        self.ground_albedo = 0.7
        self.reflection_type = 0        # lambert

        # reciver
        self.reciver_height = None
        self.reciver_angle = None

        # sun
        self.sun_intensity = 1000
        self.sun_elevation = 0          # downward
        self.sun_azimuth = None

        # atm fields
        self.swiping_height = None
        self.height_array = None
        self.absorption_coeff_array = None
        self.scattering_coeff_array = None
        self.opacity = None


    def set_reflection_type(self, reflection_type = 'lambert'):
        """Set the reflection type for the model
            Options are lambert (0) and specular (1)
        """
        if reflection_type not in [0, 1, 'lambert', 'specular']:
            raise ValueError('Use 0, 1, "lambert", "specular" as the '\
                            'refelction type!')
        self.reflection_type = 'lambert' if reflection_type == 0 else 'specular'
        self.reflection_type = reflection_type


    def set_reciver(self, height, angle):
        """Sets the position of the reciver e.g. where the rediation field is
        evaluated.

        Parameters:
            height (float): Height of the reciver (must be positive)[km].
            angle (float):  The elevation angle the recever is looking at.
                            Must be between 0 (up) and 180 (down)[deg].
        """

        if type(height) not in [int, float]:
            raise TypeError('The height must be a non-negative real number')
        if height < 0:
            raise ValueError('The height cannot be negative')
        if type(angle) not in [int, float]:
            raise TypeError('The angle must be a real number between 0 and 180')
        if angle < 0 or angle > 180:
            raise ValueError('The angle cannot be negative or greater 180')

        self.reciver_height = RT_model_1D.km2m(height)
        self.reciver_angle = angle


    def set_sun(self, intensity, angle):
        """Sets the elevation angle of the sun and the intensity.

        Parameters:
            intensity (float): Intensity of the sun (positive) [W/m2/sr/nm].
            angle (float):  The elevation angle of the sun.
                            Must be between 0 (zenith) and 90 (horizion)[deg].
        """

        if type(intensity) not in [int, float]:
            raise TypeError('The intensity must be a non-negative real number')
        if intensity < 0:
            raise ValueError('The intensity cannot be negative')
        if type(angle) not in [int, float]:
            raise TypeError('The angle must be a real number between 0 and 180')
        if angle < 0 or angle > 90:
            raise ValueError('The angle cannot be negative or greater 90')

        self.sun_intensity = intensity
        self.sun_elevation = angle


    def define_grid(self, atm_height = 200, swiping_height = 1):
        """Sets the Grid for the atmosheric parameters.

        Parameters:
            atm_height (float): Total height [km].
            step_width (float): The height of an atm layer in witch the atm
                                parameters stay constant [km].
                                swiping height

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


    def direkt_beam_intensity(self, height):
        """ DocString """
        idx, height = RT_model_1D.argclosest(RT_model_1D.km2m(height),
                                self.height_array, return_value = True)


        tau = 0
        print(np.arange(len(self.height_array)-1,idx-1,-1))
        for lvl in np.arange(len(self.height_array)-1,idx-1,-1):

            tau += (self.absorption_coeff_array[lvl] + \
                self.scattering_coeff_array[lvl]) * \
                self.swiping_height / \
                np.cos(RT_model_1D.deg2rad(self.sun_elevation))


        I_dir = self.sun_intensity * np.exp(- tau)

        return I_dir



    def get_atmoshpere_profils(self):
        """ Doc String """
        self.absorption_coeff_array = np.zeros((len(self.height_array)))
        self.scattering_coeff_array = np.zeros((len(self.height_array)))
        dens_profil, dens_profil_height = RT_model_1D.readin_densprofil()

        for idx, height in enumerate(np.nditer(self.height_array)):
            dens = dens_profil[RT_model_1D.argclosest(height,
                            RT_model_1D.km2m(dens_profil_height))]
            self.absorption_coeff_array[idx] = dens * self.absorption_cross_sec
            self.scattering_coeff_array[idx] = dens * self.scattering_cross_sec
        return self.absorption_coeff_array, self.scattering_coeff_array


    @staticmethod
    def readin_densprofil():
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

        return (1-g**2)/(1+g**2-2*g*np.cos(ang))**(3/2)


    def calc_scattering_angle(self, theta, phi = None):
        """Calculates the angle between the incoming and outgoing pencilbeam.
        It is needed for the Phasefunction. Returns the angle in deg.
        """
        if type(theta) not in [int, float]:
            raise TypeError('Theta must be a real number between -180 and 180')
        if theta < -180 and theta > 180:
            raise ValueError('Theta cannot be negative')

        delta_phi = 0.
        if phi != None and self.sun_azimuth != None:
            if type(phi) not in [int, float]:
                raise TypeError('Phi must be a real number between 0 and 360')
            if phi < 0 and phi > 360:
                raise ValueError('Theta cannot be negative or above 360')

            delta_phi = RT_model_1D.deg2rad(self.sun_azimuth - phi)

        angle = np.cos(RT_model_1D.deg2rad(theta)) * \
                np.cos(RT_model_1D.deg2rad(self.sun_elevation)) + \
                np.sin(RT_model_1D.deg2rad(theta)) * \
                np.sin(RT_model_1D.deg2rad(self.sun_elevation)) * \
                np.cos(delta_phi)

        return RT_model_1D.rad2deg( np.arccos(angle) )

    def print_testvals(self):
        """prints an attribute which must be set here. It's for testing."""
        print(self.sun_azimuth)

    @staticmethod
    def argclosest(value, array, return_value = False):
        """Returns the index in ``array`` which is closest to ``value``."""
        idx = np.abs(array - value).argmin()
        return (idx, array[idx]) if return_value else idx

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



if __name__ == '__main__':
    main()

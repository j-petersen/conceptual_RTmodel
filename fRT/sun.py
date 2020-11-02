"""
Implementation of class and methodes related to the sun.
"""
from fRT import functions as f

__all__ = ["Sun"]


class Sun(object):
    def __init__(self, ele=45, azi=180, intensity=None, wavelength=500e-9):
        self.elevation = ele
        self.azimuth = azi

        if intensity is not None:
            self.intensity = intensity
        else:
            self.intensity = f.sun_init_intensity(wavelength)

    def set_elevation(self, elevation):
        self.elevation = elevation

    def set_azimuth(self, azimuth):
        self.azimuth = azimuth

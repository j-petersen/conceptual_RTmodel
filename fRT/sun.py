"""
Implementation of class and methodes related to the sun.

"""

__all__ = ["Sun"]


class Sun(object):
    def __init__(self, intensity=1000, ele=45, azi=180):
        self.intensity = intensity
        self.elevation = ele
        self.azimuth = azi

    def set_elevation(self, elevation):
        self.elevation = elevation

    def set_azimuth(self, azimuth):
        self.azimuth = azimuth

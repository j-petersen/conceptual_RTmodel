"""
Implementation of class and methodes related to the receiver.

"""

__all__ = ["Receiver"]


class Receiver(object):
    def __init__(self, height=0, ele=45, azi=180):
        self.height = height
        self.elevation = ele
        self.azimuth = azi

    def set_height(self, height):
        self.height = height

    def set_elevation(self, elevation):
        self.elevation = elevation

    def set_azimuth(self, azimuth):
        self.azimuth = azimuth

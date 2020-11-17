import numpy as np
from pathlib import Path

__all__ = ["readin_densprofile", "readin_tempprofile"]

def readin_densprofile():
    file = Path("fRT/data/density.txt")
    with open(file, 'r') as file:
        data = np.genfromtxt(file, skip_header=6, delimiter=';')

    altitude = data[:, 1]  # Altitude
    density = (data[:, 2]) * 10 ** 6  # convert to SI units
    return altitude, density

def readin_tempprofile():
    file = Path("fRT/data/temperature.txt")
    with open(file, 'r') as file:
        data = np.genfromtxt(file, skip_header=6, delimiter=';')

    altitude = data[:, 1]  # Altitude
    temperature = (data[:, 2])
    return altitude, temperature

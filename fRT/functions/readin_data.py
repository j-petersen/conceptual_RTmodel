import numpy as np

__all__ = [
    "readin_densprofile",
    "readin_tempprofile"
    ]

def readin_densprofile():
    PATH = "/Users/jonpetersen/data/data_BA/"
    MSIS_DATEI = "MSIS/MSIS_18072300_new.txt"
    with open(PATH + MSIS_DATEI) as msis:
        MSISdata = np.genfromtxt(msis, skip_header=11)
    # 0 Height, km | 1 O, cm-3 | 2 N2, cm-3 | 3 O2, cm-3 |
    # 4 Mass_density, g/cm-3 | 5 Ar, cm-3
    MSISalt = MSISdata[:, 0]  # Altitude
    MSISdens = (
        MSISdata[:, 1] + MSISdata[:, 2] + MSISdata[:, 3] + MSISdata[:, 4]
    ) * 10 ** 6  # add desitys and convert to SI units
    return MSISalt, MSISdens


def readin_tempprofile():
    PATH = "/Users/jonpetersen/data/data_BA/"
    FILE = "T_Fit_54N_Tab.txt"
    with open(PATH + FILE) as f:
        data = np.genfromtxt(f, skip_header=11, usecols=None)

    altitude = np.empty((len(data)))
    mean_temp = np.empty((len(data)))
    for id, row in enumerate(data):
        altitude[id] = row[0]
        mean_temp[id] = np.mean(row[1:])

    return altitude, mean_temp

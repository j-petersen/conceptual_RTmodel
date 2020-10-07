def argclosest(value, array):
    '''Returns the index in ``array`` which is closest to ``value``.'''
    return np.abs(array - value).argmin()

def phasefkt(ang,g):
    return (1-g**2)/(1+g**2-2*g*np.cos(ang))**(3/2)

def read_densprofil():
    PATH = '/Users/jonpetersen/data/data_BA/'
    MSIS_DATEI = 'MSIS/MSIS_18072300_new.txt'
    msis = open(PATH + MSIS_DATEI)      # 0 Height, km | 1 O, cm-3 | 2 N2, cm-3 | 3 O2, cm-3 | 4 Mass_density, g/cm-3 | 5 Ar, cm-3
    MSISdata = np.genfromtxt(msis, skip_header=11)
    MSISalt = MSISdata[:,0]        # Altitude
    MSISdens = (MSISdata[:,1] + MSISdata[:,2] + MSISdata[:,3] + MSISdata[:,4])*10**6    # Dichten von O, N2, O2 und Ar addieren für Gesamtdichte / cm^-3, nur jeder 5. Bin
    return MSISdens, MSISalt

def atmoshpere_profils(dens_profil, dens_profil_height, absorption_cross_sec, scattering_cross_section):
    alpha_array = []#np.zeros((int(atm_height*1000/ds)))
    sigma_array = []
    for height in height_array:
        dens = dens_profil[argclosest(height, MSISalt)]
        alpha_array.append(dens*absorption_cross_sec)
        sigma_array.append(dens*scattering_cross_section)
    return alpha_array, sigma_array

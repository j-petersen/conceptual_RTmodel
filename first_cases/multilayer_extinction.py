'''
This is an easy one-layer extionction model with an source and an receiver
'''

import numpy as np
import matplotlib.pyplot as plt
import typhon as ty


def main():
    ### Settings
    #model
    ds = 1 #m

    # Source
    Solar_Intensity = 1000
    Sun_height = 1000

    # Atmosphere and Boundary
    atm_height = 200 # km
    alpha = 1e-4 # 1/m
    ground_albedo = 0.7
    ALPHA = 1e-30           # absorption cross section  - scattering cross section(6.2403548416*10**(-32))

    # Receiver
    view_angle = 180
    receiv_height = 20 # km

    # MSIS
    PATH = '/Users/jonpetersen/data/data_BA/'
    MSIS_DATEI = 'MSIS/MSIS_18072300_new.txt'
    msis = open(PATH + MSIS_DATEI)      # 0 Height, km | 1 O, cm-3 | 2 N2, cm-3 | 3 O2, cm-3 | 4 Mass_density, g/cm-3 | 5 Ar, cm-3
    MSISdata = np.genfromtxt(msis, skip_header=11)
    MSISalt = MSISdata[:,0]        # Altitude
    MSISdens = (MSISdata[:,1] + MSISdata[:,2] + MSISdata[:,3] + MSISdata[:,4])*10**6    # Dichten von O, N2, O2 und Ar addieren für Gesamtdichte / cm^-3, nur jeder 5. Bin

    rad_array = np.zeros((2,int(atm_height*1000/ds))) # down and upward radiation array with an spacing of ds
    height_array = np.linspace(0,atm_height,int(atm_height*1000/ds))


    alpha_array = []#np.zeros((int(atm_height*1000/ds)))
    for height in height_array:
        dens = MSISdens[argclosest(height, MSISalt)]
        alpha_array.append(dens*ALPHA)

    rad_array[0] = path_integartion(Solar_Intensity, int(atm_height*1000/ds), alpha_array[::-1])
    ground_value = rad_array[0,-1] * ground_albedo
    rad_array[1] = path_integartion(ground_value, int(atm_height*1000/ds), alpha_array)


    ty.plots.styles.use(["typhon", 'typhon-dark'])
    fig, ax0 = plt.subplots(ncols=1)
    plot_intensity(rad_array,height_array, ax=ax0)
    fig.savefig('./../plots/multilayer_extinction')
    plt.show()

def argclosest(value, array):
    '''Returns the index in ``array`` which is closest to ``value``.'''
    return np.abs(array - value).argmin()

def path_integartion(rad_init, size, alpha, ds = 1):
    rad_array = [rad_init]
    for id in range(1,size):
        rad_array.append(rad_array[-1] * np.exp(-alpha[id]*ds))
    return np.array(rad_array)

def plot_intensity(rad_array, height_array, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(rad_array[0][::-1], height_array, label = 'downward')
    ax.plot(rad_array[1], height_array, label = 'upward')
    ax.set_xlim(0,1000)
    #ax.set_ylim(0,10)
    ax.set_xlabel(f"Intensity / a.u.")
    ax.set_ylabel(f"Height / km")
    ax.set_title('multilayer extionction')
    ax.legend(loc='upper left')



if __name__ == "__main__":
    main()

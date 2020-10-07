'''
This should be an easy one-layer extionction model with an source and an receiver
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
    atm_height = 10 # km
    alpha = 1e-4 # 1/m
    ground_albedo = 0.7

    # Receiver
    view_angle = 180
    receiv_height = 20 # km


    rad_array = np.zeros((2,int(atm_height*1000/ds))) # down and upward radiation array with an spacing of ds
    height_array = np.linspace(0,atm_height,int(atm_height*1000/ds))


    rad_array[0] = path_integartion_homogen(Solar_Intensity, int(atm_height*1000/ds))
    ground_value = rad_array[0,-1] * ground_albedo
    rad_array[1] = path_integartion_homogen(ground_value, int(atm_height*1000/ds))

    ty.plots.styles.use(["typhon", 'typhon-dark'])
    fig, ax0 = plt.subplots(ncols=1)
    plot_intensity(rad_array, height_array, ax=ax0)
    fig.savefig('./../plots/onelayer_extinction')
    plt.show()

def path_integartion_homogen(rad_init, size, alpha = 1e-4, ds = 1):
    rad_array = [rad_init]
    for id in range(1,size):
        rad_array.append(rad_array[-1] * np.exp(-alpha*ds))
    return np.array(rad_array)

def plot_intensity(rad_array, height_array, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(rad_array[0][::-1], height_array, label = 'downward')
    ax.plot(rad_array[1], height_array, label = 'upward')
    ax.set_xlim(0,1000)
    ax.set_xlabel(f"Intensity / a.u.")
    ax.set_ylabel(f"Height / km")
    ax.set_title('homogenous layer')
    ax.legend()




if __name__ == "__main__":
    main()

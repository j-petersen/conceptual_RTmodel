import numpy as np
import typhon as ty
import matplotlib.pyplot as plt
from submodules import functions as f
from submodules import plotting_routines as pr
from RTmodel_class import RT_model_1D as RTmodel
from matplotlib.ticker import StrMethodFormatter


def model_setup():
    # Control parameters
    sun_ele, sun_azi = 45, 180   # position in sky [degree, 0° = zenith]
    sun_intensity = 1000       # intensity of the sun [W/m2sr]

    global receiver_height
    global receiver_elevation_angle
    global receiver_azimuth_angle

    model = RTmodel()   # create model instance

    model.define_grid(atm_height = 200, swiping_height = 1)

    model.set_wavelenth(500e-9)
    model.set_scattering_type('rayleigh')
    # model.set_scattering_type('henyey_greenstein')

    # model.get_atmoshperic_profiles()
    model.set_atmosheric_temp_profile()

    model.toggle_plank_radiation('off')
    model.toggle_scattering('on')

    model.set_reflection_type(1)

    model.set_sun_position(sun_intensity, sun_ele, sun_azi)
    model.set_receiver(receiver_height, receiver_elevation_angle,
                        receiver_azimuth_angle)


    return model


def test(rt):
    lvl_grid = rt.height_array

    int_grid = np.empty((len(lvl_grid)))
    source_grid = np.empty((len(lvl_grid)))

    for id, lvl in enumerate(lvl_grid):
        int_grid[id] = rt.calc_direct_beam_intensity(int(lvl))
        source_grid[id] = rt.scattering_source_term(lvl)

    ty.plots.styles.use(["typhon", 'typhon-dark'])
    fig, ax = plt.subplots(ncols=1,nrows=1)
    pr.plot_intensity(list(int_grid), list(lvl_grid/1000),
                    ax = ax, multi_way = False)
    ax.set_title(rf"Receiver: h={rt.receiver_height}, " + \
                rf"$\Theta$={(rt.receiver_elevation + 90) % 180}, " + \
                rf"$\phi$={(rt.receiver_azimuth + 180) % 360}" + '\n'\
                rf"Sun: $\Theta$={(rt.sun_elevation + 90) % 180}, " + \
                rf"$\phi$={(rt.sun_azimuth + 180) % 360}",
                fontdict = {'fontsize':12}, va='bottom')
    #fig.savefig()
    plt.show()


def plotting_radiation_height(rt):
    rad_field = rt.evaluate_radiation_field()
    height_field = rt.get_receiver_viewing_field()

    max_height = 100
    height_level = None

    fig, ax = plt.subplots()
    ax.plot(rad_field, f.m2km(height_field))
    if height_level is not None:
        ax.axhline(height_level, alpha = 0.7)

    ax.grid()

    if max_height is not None:
        ax.set_ylim(0, max_height)

    ax.set_ylabel("height / km")
    ax.set_xlabel(r"Intensity / $Wm^{-2}sr^{-1}m^{-1}$")

    # ax.set_title(rf"$T_\mathrm{{B}}$ at $\Theta$ = {zenith_angle:.0f}°")

    # Plot Tb vs Viewing angle for a specific pressure level:



def plotting_radiation_at_viewingangle(rt):
    angles = np.linspace(0, 180, 40, endpoint = True)
    receiver_field = np.empty((len(angles)))
    rt.set_sun_position(1000, float(angles[8]), 180)
    for id, angle in enumerate(angles):
        rt.set_receiver(receiver_height, float(angle), receiver_azimuth_angle)

        receiver_field[id] = rt.evaluate_radiation_field()[0]


    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    ax.plot(np.deg2rad(angles), receiver_field)
    # ax.legend(loc="upper right")
    ax.set_theta_offset(np.deg2rad(+90))
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 181, 45), ha="left")
    # ax.text(0.01, 0.75, r"$T_\mathrm{B}$", transform=ax.transAxes)
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g} K"))
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xlabel(r"Viewing angle $\Theta$")
    # ax.set_title(rf"$T_\mathrm{{B}}$ at p = {pressure_level/100:.0f} hPa")
    plt.show()


def plot_phasefunction():

    g = 0
    angles = np.linspace(0, 360, 360)
    values = np.empty((len(angles)))
    for id, angle in enumerate(angles):
        values[id] = f.rayleigh_phasematrix(float(angle), stokes_dim = 1)

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    ax.plot(np.deg2rad(angles), values)
    # ax.legend(loc="upper right")
    ax.set_theta_offset(np.deg2rad(0))
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 181, 45), ha="center")
    ax.set_thetamin(0)
    ax.set_rlabel_position(-112.5)  # Move radial labels away from plotted line
    ax.set_thetamax(360)
    ax.set_xlabel(r"scattering angle $\Theta$")
    ax.set_title("Rayleigh Phasefunction",
                fontdict = {'fontsize':16}, va='bottom')
    # ax.set_title(f"Henyey-Greenstein Phasefunction (g = {g})",
    #             fontdict = {'fontsize':16}, va='bottom')
    plt.show()



if __name__ == '__main__':
    ty.plots.styles.use(["typhon", 'typhon-dark'])

    receiver_height = 0   # height of the receiver [km]
    receiver_elevation_angle = 45  # viewing angle [degree, 180° = upward rad]
    receiver_azimuth_angle = 180


    rt = model_setup()
    test(rt)
    # plotting_radiation_height(rt)
    # plotting_radiation_at_viewingangle(rt)
    # plot_phasefunction()

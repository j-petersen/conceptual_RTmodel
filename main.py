import fRT
import numpy as np
import typhon as ty
from fRT import functions as f
import plotting_routines as pr
import matplotlib.pyplot as plt


def model_setup(sun, receiver):
    # Control parameters

    rt_model = fRT.RT_model_1D()  # create model instance

    rt_model.define_grid(atm_height=200, swiping_height=1)
    rt_model.set_stokes_dim(4)

    rt_model.set_wavelenth(500e-9)
    rt_model.set_scattering_type("rayleigh")
    # model.set_scattering_type('henyey_greenstein')

    # model.get_atmoshperic_profiles()
    rt_model.set_atmosheric_temp_profile()

    rt_model.toggle_planck_radiation("off")
    rt_model.toggle_scattering("on")

    rt_model.set_reflection_type(0.5)

    rt_model.set_sun_position(sun.elevation, sun.azimuth)
    rt_model.set_receiver(receiver.height, receiver.elevation, receiver.azimuth)

    return rt_model


def test(rt):
    lvl_grid = rt.height_array

    int_grid = np.empty((len(lvl_grid)))
    source_grid = np.empty((len(lvl_grid)))

    for id, lvl in enumerate(lvl_grid):
        int_grid[id] = rt.calc_direct_beam_intensity(int(lvl))
        source_grid[id] = rt.scattering_source_term(lvl)

    ty.plots.styles.use(["typhon", "typhon-dark"])
    fig, ax = plt.subplots(ncols=1, nrows=1)
    pr.plot_intensity(list(source_grid), list(lvl_grid / 1000), ax=ax, multi_way=False)
    # ax.set_title(rf"Receiver: h={rt.receiver_height}, " + \
    #             rf"$\Theta$={(rt.receiver_elevation + 90) % 180}, " + \
    #             rf"$\phi$={(rt.receiver_azimuth + 180) % 360}" + '\n'\
    #             rf"Sun: $\Theta$={(rt.sun_elevation + 90) % 180}, " + \
    #             rf"$\phi$={(rt.sun_azimuth + 180) % 360}",
    #             fontdict = {'fontsize':12}, va='bottom')
    # fig.savefig('plots/height_scattering_source.png', dpi=300)


def plotting_radiation_height(rt, stokes_dim = None):
    rad_field = rt.evaluate_radiation_field()[:,:]
    height_field = rt.get_receiver_viewing_field()

    max_height = 200
    height_level = None
    if not stokes_dim:
        stokes_dim = np.shape(rad_field)[1]

    labels = ['I', 'Q', 'U', 'V']
    fig, ax = plt.subplots()
    for i in range(stokes_dim):
        ax.plot(rad_field[:,i], f.m2km(height_field), label=labels[i])
    ax.plot(np.sqrt(rad_field[:,1]**2 + rad_field[:,2]**2 + rad_field[:,3]**2),
            f.m2km(height_field), label = "deg of pol")
    if height_level is not None:
        ax.axhline(height_level, alpha=0.7)

    if max_height is not None:
        ax.set_ylim(0, max_height)

    sun_theta, sun_phi = f.convert_direction(rt.sun_elevation, rt.sun_azimuth)
    rec_theta, rec_phi = f.convert_direction(rt.receiver_elevation, rt.receiver_azimuth)

    ax.set_title(rf"Receiver: h={rt.receiver_height}, " + \
                rf"$\Theta$={rec_theta}, " + \
                rf"$\phi$={np.round(rec_phi,2)}" + '\n'\
                rf"Sun: $\Theta$={sun_theta}, " + \
                rf"$\phi$={sun_phi}",
                fontdict = {'fontsize':12}, va='bottom')
    ax.set_xlabel(r"Intensity / $Wm^{-2}sr^{-1}m^{-1}$")
    ax.set_ylabel("height / km")
    ax.legend()
    # fig.savefig('plots/height_intensity_from_ground', dpi=300)


def plotting_radiation_at_viewingangle(rt, sun, receiver):
    angles = np.linspace(0, 180, 40, endpoint=True)
    receiver_field = np.empty((len(angles)))
    rt.set_sun_position(angles[8], sun.azimuth)
    for id, angle in enumerate(angles):
        rt.set_receiver(receiver.height, angle, receiver.azimuth)

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
    # fig.savefig('plots/radiation_from_viewingangle', dpi=300)


def plot_phasefunction():
    g = 0.3
    angles = np.linspace(0, 360, 360)
    values = np.empty((len(angles)))
    for id, angle in enumerate(angles):
        values[id] = f.rayleigh_phasematrix(angle, stokes_dim=1)
        # values[id] = f.henyey_greenstein_phasefunc(angle, g = g)

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
    ax.set_title("Rayleigh Phasefunction", fontdict={"fontsize": 16}, va="bottom")
    # ax.set_title(f"Henyey-Greenstein Phasefunction (g = {g})",
    #             fontdict = {'fontsize':16}, va='bottom')
    # fig.savefig('plots/rayleigh_phasefunc.png', dpi=300)


def plot_deg_of_polarization(rt, receiver):
    angles = np.linspace(0, 90, 19, endpoint=True)
    receiver_field = np.empty((len(angles),4))
    for id, angle in enumerate(angles):
        rt.set_receiver(receiver.height, receiver.elevation, receiver.azimuth+angle)
        receiver_field[id] = rt.evaluate_radiation_field()[0]

    p = f.degree_of_polarization(receiver_field)
    fig, ax = plt.subplots()
    ax.plot(angles, p)
    ax.set_xlabel(r"azimuth angle diff / deg")
    ax.set_ylabel("degree of polarization")
    fig.savefig('plots/deg_of_polarization', dpi=300)

def plot_blue_red_sky(rt, sun, receiver, scaled=False):
    wavelenghs = np.array((450, 550, 650)) * 1e-9
    angles = np.arange(45, 90, 1)
    rad_field = np.empty((len(wavelenghs), len(angles)))
    for id_wave, wavelengh in enumerate(wavelenghs):
        rt.set_wavelenth(wavelengh)
        for id_angle, angle in enumerate(angles):
            rt.set_sun_position(angle, sun.azimuth)
            rt.set_receiver(receiver.height, angle, receiver.azimuth)
            rad_field[id_wave, id_angle] = rt.evaluate_radiation_field()[0]

    colors = ["ty:chalmers-blue", "ty:max-planck", "ty:uhh-red"]
    # if scaled:
    #     rad_field = (rad_field.T/rad_field.sum(axis=1).T).T

    fig, ax = plt.subplots()
    for id, wavelengh in enumerate(wavelenghs * 1e9):
        if scaled:
            ax.plot(
                angles,
                rad_field[id] / rad_field.sum(axis=0),
                color=colors[id],
                label=str(round(wavelengh)) + "nm",
            )
        else:
            ax.plot(
                angles,
                rad_field[id],
                color=colors[id],
                label=str(round(wavelengh)) + "nm",
            )

    ax.set_ylabel(r"Intensity / $Wm^{-2}sr^{-1}m^{-1}$")
    ax.set_xlabel(r"Sun elevation $\Theta$")
    ax.legend()
    # fig.savefig('plots/scattering_wavelength_dependence.png', dpi=300)


def plot_sky_stationary_sun(rt, sun, receiver):
    angle_resolution = 10
    rt.set_sun_position(sun.elevation, sun.azimuth)
    elevations = np.arange(0, 90, angle_resolution)

    azimuths = np.arange(
        sun.azimuth - 60, sun.azimuth + 60 + angle_resolution, angle_resolution)

    wavelengths = np.array((650, 550, 450)) * 1e-9

    rad_field = np.empty((len(elevations), len(azimuths), len(wavelengths)))
    counter = 1
    for id_wave, wavelength in enumerate(wavelengths):
        rt.set_wavelenth(wavelength)
        for id_ele, ele in enumerate(elevations):
            for id_azi, azi in enumerate(azimuths):
                rt.set_receiver(0, ele, azi)
                rad_field[id_ele, id_azi, id_wave] = rt.evaluate_radiation_field()[0]
                if abs(ele - sun.elevation) <= 0.5 and abs(azi - sun.azimuth) <= 0.5:
                    rad_field[id_ele, id_azi, id_wave] = 0
                else:
                    rad_field[id_ele, id_azi, id_wave] = rt.evaluate_radiation_field()[
                        0
                    ]

                print(f"Step {counter} of {rad_field.size} finised!", end="\r")
                counter += 1
    for id_wave, wavelength in enumerate(wavelengths):
        rad_field[:, :, id_wave] = (
            rad_field[:, :, id_wave] / np.max(rad_field.sum(axis=2)) * 255 * 3
        )

    img = np.array(rad_field, dtype=int)
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation=None)

    ax.set_xticks(np.arange(len(azimuths)))
    ax.set_yticks(np.arange(len(elevations)))
    ax.set_xticklabels(azimuths)
    ax.set_yticklabels(elevations)

    ax.set_xlabel(r"azimuth $\varphi$")
    ax.set_ylabel(r"elevation $\Theta$")
    ax.set_title(rf"sun elevation = {sun.elevation}$^\circ$")

    # fig.savefig(f'plots/rgb_sky_at_{sun.elevation}_sunelevation.png', dpi=300)


if __name__ == "__main__":
    ty.plots.styles.use(["typhon", "typhon-dark"])

    receiver = fRT.Receiver(height=0, ele=45, azi=0)
    sun = fRT.Sun(ele=45, azi=0)

    model = model_setup(sun, receiver)
    # test(model)
    plotting_radiation_height(model)
    # plotting_radiation_at_viewingangle(model, sun, receiver)
    # plot_deg_of_polarization(model, receiver)
    # plot_blue_red_sky(model, sun, receiver, scaled=True)
    # plot_phasefunction()

    # elevations = np.arange(0, 90, 10)
    # for count, ele in enumerate(elevations):
    #     sun.set_elevation(ele)
    #     plot_sky_stationary_sun(model, sun, receiver)
    #     print(f"Done with {count+1} of {len(elevations)} Plots!")
    plt.show()

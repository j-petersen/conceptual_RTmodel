import fRT
import numpy as np
from scipy.linalg import expm
from fRT import functions as f

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

def testing_transformed_rayleigh():
    stokes_dim = 4
    term = (np.eye(stokes_dim) - expm(-np.eye(stokes_dim) * 1))
    stokes = np.zeros(stokes_dim)
    stokes[0] = 1
    P = f.transformed_rayleigh_scattering_matrix(
        100,
        100,
        180,
        270,
        stokes_dim = stokes_dim
    )
    if stokes_dim == 1:
        print(term * stokes * P)
    else:
        print((term @ stokes @ P)/(term @ stokes @ P)[0])

if __name__ == "__main__":
    # receiver = fRT.Receiver(height=50, ele=45, azi=0)
    # sun = fRT.Sun(ele=45, azi=0)
    #
    # model = model_setup(sun, receiver)
    testing_transformed_rayleigh()

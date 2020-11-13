import numpy as np

__all__ = [
    'degree_of_polarization'
]

def degree_of_polarization(stokes, stokes_dim=4):
    if stokes.shape == (stokes_dim,):
        p = np.sqrt(np.sum(stokes[1:] ** 2)) / stokes[0]
    else:
        p = np.sqrt(np.sum(stokes[:,1:]**2, axis=1)) / stokes[:,0]

    return p

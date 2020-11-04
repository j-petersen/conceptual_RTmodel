import numpy as np

__all__ = ["argclosest", "isclose", "delta_func"]


def argclosest(value: object, array: object, return_value: object = False) -> object:
    """Returns the index in ``array`` which is closest to ``value``."""
    idx = np.abs(array - value).argmin()
    return (idx, array[idx].item()) if return_value else idx


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """rel_tol is a relative tolerance, it is multiplied by the greater of the
    magnitudes of the two arguments; as the values get larger, so does the
    allowed difference between them while still considering them equal.
    abs_tol is an absolute tolerance that is applied as-is in all cases. If the
    difference is less than either of those tolerances, the values are
    considered equal."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def delta_func(n):
    """Delta function (numerical rounding and precision issues are considered
    at the comparioson for float equality)"""

    if isclose(n, 0):
        return 1
    else:
        return 0

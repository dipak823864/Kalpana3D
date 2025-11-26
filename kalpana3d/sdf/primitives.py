from numba import njit
import numpy as np
from kalpana3d.math.vec3 import length, dot

@njit(fastmath=True)
def sdf_capsule(p, a, b, r):
    """
    Signed Distance Function for a capsule.
    """
    pa = p - a
    ba = b - a
    # Numba compatible clipping
    h = max(0.0, min(dot(pa, ba) / dot(ba, ba), 1.0))
    return length(pa - ba * h) - r

@njit(fastmath=True)
def sdf_sphere(p, r):
    """
    Signed Distance Function for a sphere centered at the origin.

    Args:
        p: A 3D point (NumPy array).
        r: The radius of the sphere.

    Returns:
        The signed distance from point p to the sphere.
    """
    return length(p) - r

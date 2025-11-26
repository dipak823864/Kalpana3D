from numba import njit
from kalpana3d.math.vec3 import length

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

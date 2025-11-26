import numpy as np
from numba import njit

@njit(fastmath=True)
def vec3(x, y, z):
    """Creates a 3D vector."""
    return np.array([x, y, z], dtype=np.float64)

@njit(fastmath=True)
def length(v):
    """Calculates the length (magnitude) of a vector."""
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

@njit(fastmath=True)
def normalize(v):
    """Normalizes a vector to unit length."""
    l = length(v)
    if l == 0:
        return vec3(0.0, 0.0, 0.0)
    return v / l

@njit(fastmath=True)
def dot(v1, v2):
    """Calculates the dot product of two vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@njit(fastmath=True)
def cross(v1, v2):
    """Calculates the cross product of two vectors."""
    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return vec3(x, y, z)

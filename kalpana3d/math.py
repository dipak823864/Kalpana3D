
import numpy as np
from numba import njit

@njit(fastmath=True)
def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float32)

@njit(fastmath=True)
def length(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

@njit(fastmath=True)
def normalize(v):
    l = length(v)
    if l > 0:
        return v / l
    return v

@njit(fastmath=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@njit(fastmath=True)
def cross(a, b):
    return vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

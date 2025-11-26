
import numpy as np
from numba import njit
from kalpana3d.math import *

# Primitives
@njit(fastmath=True)
def sdSphere(p, radius):
    return length(p) - radius

@njit(fastmath=True)
def sdBox(p, b):
    q = np.abs(p) - b
    return length(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

@njit(fastmath=True)
def sdCylinder(p, h, r):
    d = np.abs(np.array([length(p[0:2]), p[2]])) - np.array([r, h])
    return min(max(d[0], d[1]), 0.0) + length(np.maximum(d, 0.0))

@njit(fastmath=True)
def sdCapsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = dot(pa, ba) / dot(ba, ba)
    h = max(0.0, min(h, 1.0)) # Manual clip
    return length(pa - ba * h) - r

@njit(fastmath=True)
def sdTorus(p, t):
    # t.x is main radius, t.y is tube radius
    q = np.array([length(p[0:2]) - t[0], p[2]])
    return length(q) - t[1]

# Operators
@njit(fastmath=True)
def opUnion(d1, d2):
    return min(d1, d2)

@njit(fastmath=True)
def opSubtraction(d1, d2):
    return max(-d1, d2)

@njit(fastmath=True)
def opIntersection(d1, d2):
    return max(d1, d2)

@njit(fastmath=True)
def opSmoothUnion(d1, d2, k):
    h = 0.5 + 0.5 * (d2 - d1) / k
    h = max(0.0, min(h, 1.0)) # Manual clip
    # lerp(d2, d1, h)
    m = d2 + h * (d1 - d2)
    return m - k * h * (1.0 - h)


@njit(fastmath=True)
def opWarp(p, noise_func, frequency, amplitude):
    q = vec3(noise_func(p, frequency, 1.0),
             noise_func(p + vec3(5.2, 1.3, 7.1), frequency, 1.0),
             noise_func(p + vec3(9.1, 3.7, 2.4), frequency, 1.0))
    return p + q * amplitude

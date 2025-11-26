
import numpy as np
from numba import njit, prange
from kalpana3d.math import *

# Classic Perlin noise permutation table (pre-shuffled)
p = np.array([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,
    169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
], dtype=np.int32)

# Double the permutation table to avoid buffer overflows
p_double = np.concatenate((p, p))

@njit(fastmath=True)
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@njit(fastmath=True)
def lerp(t, a, b):
    return a + t * (b - a)

@njit(fastmath=True)
def grad(hash_val, x, y, z):
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (z if h in (12, 14) else x)

    res = 0.0
    if (h & 1) != 0:
        res += u
    else:
        res -= u

    if (h & 2) != 0:
        res += v
    else:
        res -= v

    return res

@njit(fastmath=True)
def pnoise3d(p_vec, freq, amp):
    p_vec = p_vec * freq
    xi = int(np.floor(p_vec[0]))
    yi = int(np.floor(p_vec[1]))
    zi = int(np.floor(p_vec[2]))

    x = p_vec[0] - xi
    y = p_vec[1] - yi
    z = p_vec[2] - zi

    xi = xi & 255
    yi = yi & 255
    zi = zi & 255

    u = fade(x)
    v = fade(y)
    w = fade(z)

    # Hash coordinates of the 8 cube corners
    A = p_double[xi] + yi
    AA = p_double[A] + zi
    AB = p_double[A + 1] + zi
    B = p_double[xi + 1] + yi
    BA = p_double[B] + zi
    BB = p_double[B + 1] + zi

    # Add blended results from 8 corners of cube
    res = lerp(w, lerp(v, lerp(u, grad(p_double[AA], x, y, z),
                                     grad(p_double[BA], x-1, y, z)),
                             lerp(u, grad(p_double[AB], x, y-1, z),
                                     grad(p_double[BB], x-1, y-1, z))),
                     lerp(v, lerp(u, grad(p_double[AA+1], x, y, z-1),
                                     grad(p_double[BA+1], x-1, y, z-1)),
                             lerp(u, grad(p_double[AB+1], x, y-1, z-1),
                                     grad(p_double[BB+1], x-1, y-1, z-1))))

    return res * amp

from numba import njit
import numpy as np

# This implementation is adapted from the perlin-numpy library, optimized for Numba,
# and modified to compute noise at a single 3D point rather than for a full grid.
# Original source: https://github.com/pvigier/perlin-numpy

@njit(fastmath=True)
def _interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@njit(fastmath=True)
def _perlin_noise_3d(p, res, gradients):
    """
    Computes Perlin noise at a single 3D point.
    """
    p_fract = p * res % 1

    # Gradient indices
    p_int0 = np.floor(p * res).astype(np.int32)
    p_int1 = (p_int0 + 1)

    # Wrap indices to stay within the gradient table bounds
    res_x, res_y, res_z = gradients.shape[:3]
    ix0, iy0, iz0 = p_int0[0] % res_x, p_int0[1] % res_y, p_int0[2] % res_z
    ix1, iy1, iz1 = p_int1[0] % res_x, p_int1[1] % res_y, p_int1[2] % res_z

    # Fetch gradients
    g000 = gradients[ix0, iy0, iz0]
    g100 = gradients[ix1, iy0, iz0]
    g010 = gradients[ix0, iy1, iz0]
    g110 = gradients[ix1, iy1, iz0]
    g001 = gradients[ix0, iy0, iz1]
    g101 = gradients[ix1, iy0, iz1]
    g011 = gradients[ix0, iy1, iz1]
    g111 = gradients[ix1, iy1, iz1]

    # Ramps
    n000 = np.dot(p_fract - np.array([0, 0, 0]), g000)
    n100 = np.dot(p_fract - np.array([1, 0, 0]), g100)
    n010 = np.dot(p_fract - np.array([0, 1, 0]), g010)
    n110 = np.dot(p_fract - np.array([1, 1, 0]), g110)
    n001 = np.dot(p_fract - np.array([0, 0, 1]), g001)
    n101 = np.dot(p_fract - np.array([1, 0, 1]), g101)
    n011 = np.dot(p_fract - np.array([0, 1, 1]), g011)
    n111 = np.dot(p_fract - np.array([1, 1, 1]), g111)

    # Interpolation
    t = _interpolant(p_fract)
    n00 = n000 * (1 - t[0]) + t[0] * n100
    n10 = n010 * (1 - t[0]) + t[0] * n110
    n01 = n001 * (1 - t[0]) + t[0] * n101
    n11 = n011 * (1 - t[0]) + t[0] * n111
    n0 = (1 - t[1]) * n00 + t[1] * n10
    n1 = (1 - t[1]) * n01 + t[1] * n11
    return (1 - t[2]) * n0 + t[2] * n1

@njit(fastmath=True)
def fbm_noise_3d(p, res, octaves, persistence, lacunarity, gradients):
    """
    Computes Fractal Brownian Motion (FBM) noise at a single 3D point.
    """
    noise = 0.0
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * _perlin_noise_3d(p * frequency, res, gradients)
        frequency *= lacunarity
        amplitude *= persistence
    return noise

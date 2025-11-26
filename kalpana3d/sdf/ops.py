from numba import njit
import numpy as np

@njit(fastmath=True)
def op_union(d1, d2):
    """
    Standard union of two SDFs.
    """
    return min(d1, d2)

@njit(fastmath=True)
def op_smooth_union(d1, d2, k):
    """
    Smooth union of two SDFs, creating an organic blend.
    k is the smoothing factor: a smaller k creates a sharper blend.
    """
    # Manual clip implementation for Numba compatibility with scalars
    val = 0.5 + 0.5 * (d2 - d1) / k
    h = max(0.0, min(val, 1.0))
    # Linear interpolation: start + t * (end - start)
    return (d2 + h * (d1 - d2)) - k * h * (1.0 - h)

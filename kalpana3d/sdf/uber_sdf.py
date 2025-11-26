from numba import njit
import numpy as np
from kalpana3d.math.vec3 import vec3
from kalpana3d.sdf.primitives import sdf_sphere, sdf_capsule
from kalpana3d.sdf.ops import op_smooth_union
from kalpana3d.sdf.noise import fbm_noise_3d
from kalpana3d.parsers.yaml_parser import TYPE_ID_SPHERE, TYPE_ID_CAPSULE

@njit(fastmath=True)
def uber_sdf(p, object_data, global_params, gradients):
    """
    A static, "uber" SDF that processes an array of object data.
    """
    # Apply domain warping if enabled
    warp_amp = global_params[2] # domain_warp_amp
    if warp_amp > 0:
        warp_freq = global_params[1] # domain_warp_freq
        res = np.array([32, 32, 32])
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0

        displacement = vec3(
            fbm_noise_3d(p * warp_freq, res, octaves, persistence, lacunarity, gradients),
            fbm_noise_3d((p * warp_freq) + vec3(5.2, 1.3, 0.0), res, octaves, persistence, lacunarity, gradients),
            fbm_noise_3d((p * warp_freq) + vec3(0.0, 2.8, 4.5), res, octaves, persistence, lacunarity, gradients)
        ) * warp_amp
        p = p + displacement

    # Initialize with a large value
    d = 99999.0

    k = global_params[0] # smooth_union_k

    for i in range(len(object_data)):
        obj = object_data[i]
        obj_type = int(obj[0])

        if obj_type == TYPE_ID_SPHERE:
            center = vec3(obj[1], obj[2], obj[3])
            radius = obj[4]
            d_obj = sdf_sphere(p - center, radius)
            d = op_smooth_union(d, d_obj, k)

        elif obj_type == TYPE_ID_CAPSULE:
            a = vec3(obj[1], obj[2], obj[3])
            b = vec3(obj[4], obj[5], obj[6])
            radius = obj[7]
            d_obj = sdf_capsule(p, a, b, radius)
            d = op_smooth_union(d, d_obj, k)

    return d

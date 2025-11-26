
import numpy as np
from numba import njit, prange
from PIL import Image
from kalpana3d.math import *
from kalpana3d.sdf import *

MAX_STEPS = 100
MAX_DIST = 100.0
SURF_DIST = 1e-3

@njit(fastmath=True)
def get_normal(p, scene_dist_func):
    e = 0.0001
    n = vec3(
        scene_dist_func(p + vec3(e, 0, 0)) - scene_dist_func(p - vec3(e, 0, 0)),
        scene_dist_func(p + vec3(0, e, 0)) - scene_dist_func(p - vec3(0, e, 0)),
        scene_dist_func(p + vec3(0, 0, e)) - scene_dist_func(p - vec3(0, 0, e))
    )
    return normalize(n)

@njit(fastmath=True)
def get_light(p, normal, light_pos):
    light_dir = normalize(light_pos - p)
    diffuse = dot(normal, light_dir)
    diffuse = max(0.1, diffuse) # Ambient light
    return diffuse

@njit(fastmath=True)
def ray_march(ro, rd, scene_dist_func):
    dO = 0.0
    for i in range(MAX_STEPS):
        p = ro + rd * dO
        dS = scene_dist_func(p)
        if dS < SURF_DIST:
            return dO
        dO += dS
        if dO > MAX_DIST:
            break
    return 0.0

@njit(fastmath=True, parallel=True)
def render_internal(pixels, width, height, camera_pos, camera_target, light_pos, scene_dist_func):
    # Camera setup
    forward = normalize(camera_target - camera_pos)
    right = normalize(cross(forward, vec3(0, 1, 0)))
    up = normalize(cross(right, forward))

    for i in prange(width):
        for j in prange(height):
            u = (2.0 * i - width) / height
            v = (2.0 * j - height) / height

            rd = normalize(u * right + v * up + 1.5 * forward)

            dist = ray_march(camera_pos, rd, scene_dist_func)

            if dist > 0.0:
                p = camera_pos + rd * dist
                normal = get_normal(p, scene_dist_func)
                diffuse = get_light(p, normal, light_pos)
                pixels[j, i] = int(diffuse * 255)

def render(width, height, camera_pos, camera_target, light_pos, scene_dist_func, filename):
    pixels = np.zeros((height, width), dtype=np.uint8)

    # Numba can't take a function as an argument if it's not a njit function.
    # So we need to wrap the scene_dist_func in a njit function.
    # This is a bit of a hack, but it's the only way to do it.

    # A simple trick to pass a dynamic function to a Numba JIT function
    # is to have the outer Python function return the inner JIT function.
    # However, for this project, the scene function will be defined in the test script
    # and passed to render. The test script itself will handle the @njit decoration.

    render_internal(pixels, width, height, camera_pos, camera_target, light_pos, scene_dist_func)

    # Pillow expects (width, height)
    img = Image.fromarray(pixels.T, 'L')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(filename)

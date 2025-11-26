import numpy as np
from numba import njit, prange
from PIL import Image
from kalpana3d.math.vec3 import vec3, normalize, dot
from kalpana3d.sdf.primitives import sdf_sphere

@njit(fastmath=True)
def get_normal(p, scene_sdf):
    """
    Calculates the surface normal at a point p using the SDF gradient.
    """
    epsilon = 0.0001
    dx = scene_sdf(p + vec3(epsilon, 0, 0)) - scene_sdf(p - vec3(epsilon, 0, 0))
    dy = scene_sdf(p + vec3(0, epsilon, 0)) - scene_sdf(p - vec3(0, epsilon, 0))
    dz = scene_sdf(p + vec3(0, 0, epsilon)) - scene_sdf(p - vec3(0, 0, epsilon))
    return normalize(vec3(dx, dy, dz))

@njit(fastmath=True)
def ray_march(ro, rd, scene_sdf, max_steps=100, min_dist=0.001, max_dist=100.0):
    """
    Performs the ray marching algorithm.
    """
    t = 0.0
    for i in range(max_steps):
        p = ro + t * rd
        dist = scene_sdf(p)
        if dist < min_dist:
            return t, True
        if t > max_dist:
            break
        t += dist
    return t, False

@njit(parallel=True, fastmath=True)
def render(width, height, camera_pos, light_pos, scene_sdf):
    """
    Renders the scene using ray marching.
    """
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    aspect_ratio = width / height

    for y in prange(height):
        for x in range(width):
            # Screen coordinates
            u = (2.0 * x - width) / width * aspect_ratio
            v = -(2.0 * y - height) / height

            # Ray setup
            ro = camera_pos
            rd = normalize(vec3(u, v, -1.0) - ro)

            # Ray march
            dist, hit = ray_march(ro, rd, scene_sdf)

            if hit:
                # Intersection point
                p = ro + dist * rd

                # Lighting
                normal = get_normal(p, scene_sdf)
                light_dir = normalize(light_pos - p)
                diffuse = max(0.0, dot(normal, light_dir))

                color = int(diffuse * 255)
                pixels[y, x] = [color, color, color]
            else:
                pixels[y, x] = [0, 0, 0] # Background color

    return pixels

def save_image(pixels, filepath):
    """
    Saves the rendered pixel array to a PNG file.
    """
    image = Image.fromarray(pixels, 'RGB')
    image.save(filepath)

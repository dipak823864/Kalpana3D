
import numpy as np
from numba import njit
from kalpana3d.engine import render
from kalpana3d.sdf import *
from kalpana3d.math import *

@njit(fastmath=True)
def get_scene_dist(p):
    # Capsule parameters
    cap_a = vec3(-1.0, -0.5, 0.0)
    cap_b = vec3(1.0, 0.5, 0.0)
    cap_r = 0.25

    # Torus parameters
    torus_t = np.array([1.0, 0.25])

    # Create shapes
    capsule = sdCapsule(p, cap_a, cap_b, cap_r)
    torus = sdTorus(p - vec3(0.0, 0.0, 1.0), torus_t)

    # Combine shapes
    return opUnion(capsule, torus)

def main():
    width = 800
    height = 600
    camera_pos = vec3(2.5, 2.5, 2.5)
    camera_target = vec3(0, 0, 0)
    light_pos = vec3(2, 3, 2)
    output_filename = "gallery/images/01_shapes.png"

    print("Rendering scene...")
    render(width, height, camera_pos, camera_target, light_pos, get_scene_dist, output_filename)
    print(f"Image saved to {output_filename}")

if __name__ == '__main__':
    main()

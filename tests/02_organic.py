
import numpy as np
from numba import njit
from kalpana3d.engine import render
from kalpana3d.sdf import *
from kalpana3d.math import *
from kalpana3d.noise import pnoise3d

@njit(fastmath=True)
def get_scene_dist(p):
    # Warp the domain
    p = opWarp(p, pnoise3d, 10.0, 0.1)

    # Define two spheres
    sphere1 = sdSphere(p - vec3(-0.5, 0, 0), 0.75)
    sphere2 = sdSphere(p - vec3(0.5, 0, 0), 0.75)

    # Blend them smoothly
    return opSmoothUnion(sphere1, sphere2, 0.5)

def main():
    width = 800
    height = 600
    camera_pos = vec3(0, 0, 3)
    camera_target = vec3(0, 0, 0)
    light_pos = vec3(2, 3, 2)
    output_filename = "gallery/images/02_organic.png"

    print("Rendering scene...")
    render(width, height, camera_pos, camera_target, light_pos, get_scene_dist, output_filename)
    print(f"Image saved to {output_filename}")

if __name__ == '__main__':
    main()

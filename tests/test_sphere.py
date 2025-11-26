import time
from kalpana3d.render.raymarcher import render, save_image
from kalpana3d.math.vec3 import vec3
from kalpana3d.sdf.primitives import sdf_sphere
from numba import njit

@njit(fastmath=True)
def scene_sdf(p):
    return sdf_sphere(p, 1.0)

def main():
    """
    Main function to render the sphere test.
    """
    width = 800
    height = 600

    camera_pos = vec3(0.0, 0.0, 5.0)
    light_pos = vec3(4.0, 4.0, 5.0)

    output_filepath = "gallery/sphere_test.png"

    print("Starting render...")
    start_time = time.time()

    # JIT compilation can take a moment on the first run
    pixels = render(width, height, camera_pos, light_pos, scene_sdf)

    end_time = time.time()
    print(f"Render finished in {end_time - start_time:.2f} seconds.")

    print(f"Saving image to {output_filepath}...")
    save_image(pixels, output_filepath)
    print("Image saved.")

if __name__ == "__main__":
    main()

import time
from numba import njit
from kalpana3d.mesher.marching_cubes import export_obj
from kalpana3d.sdf.primitives import sdf_sphere

# The SDF function must be defined at the top level for Numba's JIT compilation.
@njit(fastmath=True)
def sphere_scene(p):
    return sdf_sphere(p, 1.0)

def main():
    """
    Main function to test the Marching Cubes mesher.
    """
    output_filepath = "gallery/sphere.obj"
    bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    resolution = 32

    print("Starting mesh generation...")
    start_time = time.time()

    export_obj(output_filepath, sphere_scene, bounds, resolution)

    end_time = time.time()
    print(f"Meshing finished in {end_time - start_time:.2f} seconds.")
    print(f"Mesh saved to {output_filepath}")

if __name__ == "__main__":
    main()

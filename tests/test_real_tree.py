import time
import numpy as np

from kalpana3d.scene import load_scene
from kalpana3d.render.raymarcher import render, save_image
from kalpana3d.mesher.marching_cubes import export_obj
from kalpana3d.math.vec3 import vec3

def main():
    """
    Main function to run the final demonstration:
    1. Parse the YAML scene.
    2. Render verification images (close-up and full view).
    3. Export a high-resolution 3D model.
    """
    yaml_filepath = "examples/twisted_tree.yaml"

    print(f"Loading scene from {yaml_filepath}...")
    try:
        # 1. Load the scene, which creates the SDF
        scene = load_scene(yaml_filepath)
        scene_sdf = scene.sdf
    except Exception as e:
        print(f"Error loading scene: {e}")
        return

    # --- 2. Render Verification Images ---
    width, height = 400, 300
    light_pos = vec3(4.0, 5.0, 5.0)

    # --- Image A: Detail View ---
    print("Rendering detail view (tree_detail.png)...")
    camera_pos_detail = vec3(0.0, 1.0, 2.0)

    start_time = time.time()
    pixels_detail = render(width, height, camera_pos_detail, light_pos, scene_sdf)
    end_time = time.time()
    print(f"Detail render finished in {end_time - start_time:.2f} seconds.")

    save_image(pixels_detail, "gallery/tree_detail.png")
    print("Detail image saved to gallery/tree_detail.png")

    # --- Image B: Full View ---
    print("\nRendering full view (tree_full_view.png)...")
    camera_pos_full = vec3(0.0, 1.0, 7.0)

    start_time = time.time()
    pixels_full = render(width, height, camera_pos_full, light_pos, scene_sdf)
    end_time = time.time()
    print(f"Full view render finished in {end_time - start_time:.2f} seconds.")

    save_image(pixels_full, "gallery/tree_full_view.png")
    print("Full view image saved to gallery/tree_full_view.png")


    # --- 3. Export the 3D Model ---
    print("\nExporting 3D model...")
    obj_filepath = "gallery/real_tree.obj"
    bounds = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    resolution = 32

    start_time = time.time()
    export_obj(obj_filepath, scene_sdf, bounds, resolution)
    end_time = time.time()

    print(f"Meshing finished in {end_time - start_time:.2f} seconds.")
    print(f"Model saved to {obj_filepath}")


if __name__ == "__main__":
    main()

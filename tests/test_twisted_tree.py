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
    2. Render a preview image.
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

    # --- 2. Render Detail View (Close-up) ---
    print("Rendering detail view...")
    img_filepath_detail = "gallery/tree_detail.png"
    width, height = 800, 600
    camera_pos_detail = vec3(0.0, 1.0, 2.0)  # Close camera
    light_pos = vec3(4.0, 5.0, 5.0)

    start_time = time.time()
    pixels_detail = render(width, height, camera_pos_detail, light_pos, scene_sdf)
    end_time = time.time()
    print(f"Detail render finished in {end_time - start_time:.2f} seconds.")

    save_image(pixels_detail, img_filepath_detail)
    print(f"Detail image saved to {img_filepath_detail}")

    # --- 3. Render Full View (Far) ---
    print("\nRendering full view...")
    img_filepath_full = "gallery/tree_full_view.png"
    camera_pos_full = vec3(0.0, 1.0, 8.0)  # Far camera

    start_time = time.time()
    pixels_full = render(width, height, camera_pos_full, light_pos, scene_sdf)
    end_time = time.time()
    print(f"Full view render finished in {end_time - start_time:.2f} seconds.")

    save_image(pixels_full, img_filepath_full)
    print(f"Full view image saved to {img_filepath_full}")

    # --- 4. Export the 3D Model ---
    print("\nExporting 3D model...")
    obj_filepath = "gallery/twisted_tree.obj"
    bounds = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    resolution = 64 # Higher resolution for the final model

    start_time = time.time()
    export_obj(obj_filepath, scene_sdf, bounds, resolution)
    end_time = time.time()

    print(f"Meshing finished in {end_time - start_time:.2f} seconds.")
    print(f"Model saved to {obj_filepath}")


if __name__ == "__main__":
    main()

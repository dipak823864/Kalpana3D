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

    # --- 2. Render a Preview Image ---
    print("Rendering preview image...")
    img_filepath = "gallery/real_tree_fix.png"
    width, height = 800, 600
    camera_pos = vec3(0.0, 1.0, 5.0)
    light_pos = vec3(4.0, 5.0, 5.0)

    start_time = time.time()
    pixels = render(width, height, camera_pos, light_pos, scene_sdf)
    end_time = time.time()
    print(f"Render finished in {end_time - start_time:.2f} seconds.")

    save_image(pixels, img_filepath)
    print(f"Preview image saved to {img_filepath}")

    # --- 3. Export the 3D Model ---
    print("\nExporting 3D model...")
    obj_filepath = "gallery/real_tree_fix.obj"
    bounds = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    resolution = 64 # Higher resolution for the final model

    start_time = time.time()
    export_obj(obj_filepath, scene_sdf, bounds, resolution)
    end_time = time.time()

    print(f"Meshing finished in {end_time - start_time:.2f} seconds.")
    print(f"Model saved to {obj_filepath}")


if __name__ == "__main__":
    main()

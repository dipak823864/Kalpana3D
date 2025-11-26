import time
import numpy as np

from kalpana3d.parsers.yaml_parser import parse_scene_to_sdf
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

    print(f"Parsing scene from {yaml_filepath}...")
    try:
        # 1. Parse the scene and get the dynamically compiled SDF
        scene_sdf = parse_scene_to_sdf(yaml_filepath)
    except Exception as e:
        print(f"Error parsing scene: {e}")
        return

    # --- 2. Render a Preview Image ---
    print("Rendering preview image...")
    img_filepath = "gallery/twisted_tree_preview.png"
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

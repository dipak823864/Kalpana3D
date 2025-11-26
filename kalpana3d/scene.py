import numpy as np
from numba import njit, prange
from kalpana3d.parsers.yaml_parser import parse_scene_to_array
from kalpana3d.sdf.uber_sdf import uber_sdf

class Scene:
    """
    Represents a 3D scene, including the SDF defining the geometry
    and camera settings.
    """
    def __init__(self, sdf_func):
        self.sdf = sdf_func

class Camera:
    """
    Defines the viewpoint and projection settings for rendering a scene.
    """
    def __init__(self, resolution=(1024, 1024), position=None, look_at=None, fov=45.0):
        self.resolution = resolution
        self.position = np.array(position) if position else np.array([0.0, 0.0, 5.0])
        self.look_at = np.array(look_at) if look_at else np.array([0.0, 0.0, 0.0])
        self.fov = fov
        self.up = np.array([0.0, 1.0, 0.0]) # Standard up vector

def load_scene(filepath):
    """
    Loads a scene from a YAML file, creating the SDF and setting up the camera.
    """
    object_data, global_params_dict = parse_scene_to_array(filepath)

    # Pack global params into a NumPy array for Numba compatibility
    global_params = np.array([
        global_params_dict['smooth_union_k'],
        global_params_dict['domain_warp_freq'],
        global_params_dict['domain_warp_amp']
    ], dtype=np.float64)

    # Create a gradient table for noise functions
    res = (32, 32, 32)
    theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = np.stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3
    )

    # Curry the scene data into the final SDF function
    @njit(fastmath=True, parallel=True)
    def final_sdf(points):
        results = np.empty(len(points))
        for i in prange(len(points)):
            results[i] = uber_sdf(points[i], object_data, global_params, gradients)
        return results

    return Scene(final_sdf)

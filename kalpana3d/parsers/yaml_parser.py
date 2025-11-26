import yaml
from numba import njit
import numpy as np

from kalpana3d.math.vec3 import vec3
from kalpana3d.sdf.primitives import sdf_sphere
from kalpana3d.sdf.ops import op_smooth_union

# Collection of SDF functions that can be referenced by the parser
SDF_FUNCTIONS = {
    'sphere': sdf_sphere
}

# Collection of SDF operation functions
SDF_OPERATIONS = {
    'smooth_union': op_smooth_union
}

def _create_sdf_from_dict(scene_dict):
    """
    Recursively builds a single SDF function from a scene dictionary.
    """
    obj_type = scene_dict.get('type')

    if not obj_type:
        raise ValueError("Scene object must have a 'type' defined.")

    # Handle transformations
    # IMPORTANT: Transformations are applied in reverse order to the point (p)
    # This is more efficient than transforming the object itself.
    p_transformations = []
    if 'translate' in scene_dict:
        t = scene_dict['translate']
        p_transformations.append(f"p - vec3({t[0]}, {t[1]}, {t[2]})")

    if 'rotate' in scene_dict:
        # Note: Implementing full rotation is complex.
        # This is a simplified placeholder. A real implementation would need rotation matrices.
        pass

    # Build the transformed point string
    p_str = "p"
    if p_transformations:
        p_str = " ".join(p_transformations)

    # Handle SDF Primitives
    if obj_type in SDF_FUNCTIONS:
        func_name = SDF_FUNCTIONS[obj_type].__name__
        params = scene_dict.get('params', [])
        param_str = ", ".join(map(str, params))
        return f"{func_name}({p_str}, {param_str})"

    # Handle SDF Operations (like blending)
    elif obj_type in SDF_OPERATIONS:
        op_func_name = SDF_OPERATIONS[obj_type].__name__
        children = scene_dict.get('children', [])
        if not children or len(children) < 2:
            raise ValueError(f"'{obj_type}' operation requires at least 2 children.")

        # Recursively build SDFs for children
        child_sdfs = [_create_sdf_from_dict(child) for child in children]

        # Combine children using the operation
        # For smooth_union, extra parameter 'k' is needed
        k = scene_dict.get('k', 0.2)

        # Chain the operations: op(sdf1, op(sdf2, sdf3, k), k)
        sdf_expr = child_sdfs[0]
        for i in range(1, len(child_sdfs)):
            sdf_expr = f"{op_func_name}({sdf_expr}, {child_sdfs[i]}, {k})"

        return sdf_expr

    else:
        raise ValueError(f"Unknown object type: {obj_type}")


def parse_scene_to_sdf(filepath):
    """
    Parses a YAML scene file and returns a dynamic, JIT-compiled SDF function.

    Args:
        filepath: Path to the YAML file.

    Returns:
        A Numba-jitted function that computes the SDF for the entire scene.
    """
    with open(filepath, 'r') as f:
        scene_config = yaml.safe_load(f)

    sdf_body = _create_sdf_from_dict(scene_config)

    # Dynamically create the full function code
    # This is a powerful but potentially risky technique. It's used here to allow
    # Numba to JIT-compile the entire, dynamically generated scene SDF into a
    # single, highly efficient function.
    func_code = f"""
from numba import njit
import numpy as np
from kalpana3d.math.vec3 import vec3
from kalpana3d.sdf.primitives import sdf_sphere
from kalpana3d.sdf.ops import op_smooth_union

@njit(fastmath=True)
def scene_sdf(p):
    return {sdf_body}
"""

    # Use exec to define the function in a controlled scope
    scope = {}
    exec(func_code, globals(), scope)

    return scope['scene_sdf']

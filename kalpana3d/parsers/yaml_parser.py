import yaml
import numpy as np
from kalpana3d.math.vec3 import vec3

# Type IDs for SDF primitives
TYPE_ID_SPHERE = 1
TYPE_ID_CAPSULE = 2

def _apply_transform(point, translation):
    """Applies translation to a point."""
    if translation is not None:
        return np.array(point) + np.array(translation)
    return np.array(point)

def _parse_recursive(scene_dict, object_list, transform_stack):
    """
    Recursively traverses the scene graph to build a flat list of primitives.
    """
    obj_type = scene_dict.get('type')

    current_translation = scene_dict.get('translate', None)
    transform_stack.append(current_translation)

    if obj_type == 'smooth_union':
        children = scene_dict.get('children', [])
        for child in children:
            _parse_recursive(child, object_list, transform_stack)

    elif obj_type == 'domain_warp':
        child = scene_dict.get('child')
        if child:
            _parse_recursive(child, object_list, transform_stack)

    else:
        # Calculate the cumulative translation
        total_translation = np.array([0.0, 0.0, 0.0])
        for t in transform_stack:
            if t is not None:
                total_translation += np.array(t)

        # Pack the primitive data into a standardized format
        if obj_type == 'sphere':
            # Data: [type_id, center.x, center.y, center.z, radius, 0, 0, 0]
            center = _apply_transform([0,0,0], total_translation)
            radius = scene_dict['params'][0]
            object_list.append([TYPE_ID_SPHERE, center[0], center[1], center[2], radius, 0, 0, 0])

        elif obj_type == 'capsule':
            # Data: [type_id, a.x, a.y, a.z, b.x, b.y, b.z, radius]
            start_point = _apply_transform(scene_dict['start_point'], total_translation)
            end_point = _apply_transform(scene_dict['end_point'], total_translation)
            radius = scene_dict['radius']
            object_list.append([TYPE_ID_CAPSULE] + list(start_point) + list(end_point) + [radius])

    transform_stack.pop()

def parse_scene_to_array(filepath):
    """
    Parses a YAML scene file and returns a NumPy array of object data
    and a dictionary of global scene parameters.
    """
    with open(filepath, 'r') as f:
        scene_config = yaml.safe_load(f)

    object_list = []
    _parse_recursive(scene_config, object_list, [])

    # Global parameters (e.g., for smooth union and domain warp)
    global_params = {
        'smooth_union_k': scene_config.get('k', 0.2),
        'domain_warp_freq': 0.0,
        'domain_warp_amp': 0.0,
    }

    # Check for domain warp and extract its parameters
    def find_domain_warp(d):
        if d.get('type') == 'domain_warp':
            return d
        if 'children' in d:
            for child in d['children']:
                res = find_domain_warp(child)
                if res:
                    return res
        if 'child' in d:
            return find_domain_warp(d['child'])
        return None

    domain_warp_node = find_domain_warp(scene_config)
    if domain_warp_node:
        global_params['domain_warp_freq'] = domain_warp_node.get('frequency', 1.0)
        global_params['domain_warp_amp'] = domain_warp_node.get('amplitude', 0.1)

    return np.array(object_list, dtype=np.float64), global_params

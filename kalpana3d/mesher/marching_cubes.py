import numpy as np
from numba import njit, prange

# These are the standard Marching Cubes lookup tables.
# EDGE_TABLE maps a cube index to a 12-bit mask indicating which edges have vertices.
# TRI_TABLE maps a cube index to the vertex indices for the triangles.
from kalpana3d.mesher._mc_tables import EDGE_TABLE, TRI_TABLE

@njit(fastmath=True)
def _interpolate_vertex(p1, p2, val1, val2):
    """
    Linearly interpolates the position of the surface-crossing vertex on an edge.
    The surface is assumed to be at the iso-level 0.
    """
    if abs(val1 - val2) < 1e-6:
        return (p1 + p2) / 2.0
    mu = (0.0 - val1) / (val2 - val1)
    return p1 + mu * (p2 - p1)

@njit(fastmath=True)
def march_cubes(sdf_func, bounds, resolution):
    """
    Executes the Marching Cubes algorithm.

    Args:
        sdf_func: The Signed Distance Function defining the surface.
        bounds: A tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z)).
        resolution: The number of cubes along the longest axis.

    Returns:
        A tuple of (vertices, faces) as NumPy arrays.
    """
    min_bound = np.array(bounds[0], dtype=np.float64)
    max_bound = np.array(bounds[1], dtype=np.float64)

    size = max_bound - min_bound
    cell_size = np.max(size) / resolution

    res_x = int(size[0] / cell_size) + 1
    res_y = int(size[1] / cell_size) + 1
    res_z = int(size[2] / cell_size) + 1

    # Sample SDF on the grid
    grid_values = np.zeros((res_x, res_y, res_z), dtype=np.float64)
    for i in prange(res_x):
        for j in range(res_y):
            for k in range(res_z):
                p = min_bound + np.array([i, j, k]) * cell_size
                grid_values[i, j, k] = sdf_func(p)

    vertices = []
    faces = []

    # Process each cube
    for i in range(res_x - 1):
        for j in range(res_y - 1):
            for k in range(res_z - 1):
                # 8 corners of the cube
                cube_corners_pos = [
                    min_bound + np.array([i, j, k]) * cell_size,
                    min_bound + np.array([i + 1, j, k]) * cell_size,
                    min_bound + np.array([i + 1, j, k + 1]) * cell_size,
                    min_bound + np.array([i, j, k + 1]) * cell_size,
                    min_bound + np.array([i, j + 1, k]) * cell_size,
                    min_bound + np.array([i + 1, j + 1, k]) * cell_size,
                    min_bound + np.array([i + 1, j + 1, k + 1]) * cell_size,
                    min_bound + np.array([i, j + 1, k + 1]) * cell_size
                ]

                cube_corners_val = [
                    grid_values[i, j, k],
                    grid_values[i + 1, j, k],
                    grid_values[i + 1, j, k + 1],
                    grid_values[i, j, k + 1],
                    grid_values[i, j + 1, k],
                    grid_values[i + 1, j + 1, k],
                    grid_values[i + 1, j + 1, k + 1],
                    grid_values[i, j + 1, k + 1]
                ]

                # Determine cube index
                cube_index = 0
                for v_idx in range(8):
                    if cube_corners_val[v_idx] < 0:
                        cube_index |= (1 << v_idx)

                # Skip empty or full cubes
                if EDGE_TABLE[cube_index] == 0:
                    continue

                # Interpolate vertices on the edges
                edge_vertices = np.zeros((12, 3), dtype=np.float64)
                if EDGE_TABLE[cube_index] & 1:
                    edge_vertices[0] = _interpolate_vertex(cube_corners_pos[0], cube_corners_pos[1], cube_corners_val[0], cube_corners_val[1])
                if EDGE_TABLE[cube_index] & 2:
                    edge_vertices[1] = _interpolate_vertex(cube_corners_pos[1], cube_corners_pos[2], cube_corners_val[1], cube_corners_val[2])
                if EDGE_TABLE[cube_index] & 4:
                    edge_vertices[2] = _interpolate_vertex(cube_corners_pos[2], cube_corners_pos[3], cube_corners_val[2], cube_corners_val[3])
                if EDGE_TABLE[cube_index] & 8:
                    edge_vertices[3] = _interpolate_vertex(cube_corners_pos[3], cube_corners_pos[0], cube_corners_val[3], cube_corners_val[0])
                if EDGE_TABLE[cube_index] & 16:
                    edge_vertices[4] = _interpolate_vertex(cube_corners_pos[4], cube_corners_pos[5], cube_corners_val[4], cube_corners_val[5])
                if EDGE_TABLE[cube_index] & 32:
                    edge_vertices[5] = _interpolate_vertex(cube_corners_pos[5], cube_corners_pos[6], cube_corners_val[5], cube_corners_val[6])
                if EDGE_TABLE[cube_index] & 64:
                    edge_vertices[6] = _interpolate_vertex(cube_corners_pos[6], cube_corners_pos[7], cube_corners_val[6], cube_corners_val[7])
                if EDGE_TABLE[cube_index] & 128:
                    edge_vertices[7] = _interpolate_vertex(cube_corners_pos[7], cube_corners_pos[4], cube_corners_val[7], cube_corners_val[4])
                if EDGE_TABLE[cube_index] & 256:
                    edge_vertices[8] = _interpolate_vertex(cube_corners_pos[0], cube_corners_pos[4], cube_corners_val[0], cube_corners_val[4])
                if EDGE_TABLE[cube_index] & 512:
                    edge_vertices[9] = _interpolate_vertex(cube_corners_pos[1], cube_corners_pos[5], cube_corners_val[1], cube_corners_val[5])
                if EDGE_TABLE[cube_index] & 1024:
                    edge_vertices[10] = _interpolate_vertex(cube_corners_pos[2], cube_corners_pos[6], cube_corners_val[2], cube_corners_val[6])
                if EDGE_TABLE[cube_index] & 2048:
                    edge_vertices[11] = _interpolate_vertex(cube_corners_pos[3], cube_corners_pos[7], cube_corners_val[3], cube_corners_val[7])

                # Create triangles
                tri_table_entry = TRI_TABLE[cube_index]
                for tri_idx in range(0, len(tri_table_entry), 3):
                    if tri_table_entry[tri_idx] == -1:
                        break

                    v1_idx = len(vertices)
                    v2_idx = len(vertices) + 1
                    v3_idx = len(vertices) + 2

                    # Convert numpy arrays to tuples before appending
                    v1 = edge_vertices[tri_table_entry[tri_idx]]
                    v2 = edge_vertices[tri_table_entry[tri_idx + 1]]
                    v3 = edge_vertices[tri_table_entry[tri_idx + 2]]
                    vertices.append((v1[0], v1[1], v1[2]))
                    vertices.append((v2[0], v2[1], v2[2]))
                    vertices.append((v3[0], v3[1], v3[2]))

                    faces.append((v1_idx, v2_idx, v3_idx))

    return np.array(vertices), np.array(faces)


def export_obj(filepath, sdf_func, bounds, resolution):
    """
    Generates a mesh using Marching Cubes and exports it to an .obj file.

    Args:
        filepath: The path to save the .obj file.
        sdf_func: The Signed Distance Function defining the surface.
        bounds: A tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z)).
        resolution: The number of cubes along the longest axis.
    """
    vertices, faces = march_cubes(sdf_func, bounds, resolution)

    if vertices.size == 0:
        print("Warning: No mesh generated. The SDF might not cross the zero-level within the given bounds.")
        return

    with open(filepath, 'w') as f:
        f.write("# Kalpana3D OBJ Exporter\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces
        # .obj files use 1-based indexing
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

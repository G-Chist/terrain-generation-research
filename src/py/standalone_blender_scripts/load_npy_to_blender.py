import numpy as np
import bpy
import os


def grid_to_xyz(z_grid, start_coordinate, end_coordinate):
    """
    Convert a 2D grid of z-values into a (N*M)x3 array of [x, y, z] coordinates,
    where x and y are linearly spaced between start_coordinate and end_coordinate.
    """
    nrows, ncols = z_grid.shape
    x = np.linspace(start_coordinate, end_coordinate, ncols)
    y = np.linspace(start_coordinate, end_coordinate, nrows)
    xx, yy = np.meshgrid(x, y)
    xyz = np.column_stack((xx.ravel(), yy.ravel(), z_grid.ravel()))

    return xyz


def generate_faces_from_grid(n_row, n_col):
    """
    Generate triangle face indices for a (n_row x n_col) grid of vertices,
    assuming the vertices are flattened row-major (C-style) to a 1D array.
    """
    i = np.arange(n_row - 1)
    j = np.arange(n_col - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()
    top_left = ii + jj * n_col
    bottom_left = ii + (jj + 1) * n_col
    top_right = (ii + 1) + jj * n_col
    bottom_right = (ii + 1) + (jj + 1) * n_col
    tri1 = np.stack([top_left, bottom_left, top_right], axis=1)
    tri2 = np.stack([top_right, bottom_left, bottom_right], axis=1)
    faces = np.concatenate([tri1, tri2], axis=0)

    return faces


if __name__ == "__main__":
    npy_path = r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\wmterrain.npy"

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f".npy file not found at: {npy_path}")

    # Load heightmap from .npy
    z_grid = np.load(npy_path)  # shape (N, M)
    nrows, ncols = z_grid.shape

    # Convert to 3D vertices using helper function
    vertices = grid_to_xyz(z_grid, start_coordinate=-1.0, end_coordinate=1.0)

    # Generate faces assuming row-major grid structure
    faces = generate_faces_from_grid(nrows, ncols)

    # Create mesh in Blender
    mesh = bpy.data.meshes.new("npy_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    obj = bpy.data.objects.new("npy_terrain", mesh)

    # Add to a new collection
    collection = bpy.data.collections.new("terrain_collection")
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(obj)

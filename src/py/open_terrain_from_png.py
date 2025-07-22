import imageio.v3 as iio
import numpy as np
import bpy
import os


def load_bw_image_as_normalized_array(filepath):
    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")

    img = iio.imread(filepath)

    # Convert to float32 for normalization
    arr = img.astype(np.float32)
    min_val = arr.min()
    max_val = arr.max()

    if max_val - min_val == 0:
        arr_normalized = np.zeros_like(arr)
    else:
        arr_normalized = (arr - min_val) / (max_val - min_val)

    return arr_normalized


def grid_to_xyz(z_grid, start_coordinate, end_coordinate):
    """
    Convert a 2D grid of z-values into a (N*M)x3 array of [x, y, z] coordinates,
    where x and y are linearly spaced between start_coordinate and end_coordinate.

    Parameters:
    - z_grid: 2D NumPy array of shape (N, M), representing z-values over a grid.
    - start_coordinate: float, the starting coordinate value for both x and y axes.
    - end_coordinate: float, the ending coordinate value for both x and y axes.

    Returns:
    - xyz: NumPy array of shape (N*M, 3), where each row is [x, y, z].
    """

    # Get the number of rows (N) and columns (M) from the shape of the z_grid
    nrows, ncols = z_grid.shape

    # Create a 1D array of x coordinates (length M) evenly spaced from start to end
    x = np.linspace(start_coordinate, end_coordinate, ncols)

    # Create a 1D array of y coordinates (length N) evenly spaced from start to end
    y = np.linspace(start_coordinate, end_coordinate, nrows)

    # Create a 2D meshgrid from the x and y coordinate vectors
    # xx has shape (N, M), each row is a copy of x
    # yy has shape (N, M), each column is a copy of y
    xx, yy = np.meshgrid(x, y)

    # Flatten xx, yy, and z_grid into 1D arrays (length N*M each),
    # and stack them as columns to form an (N*M)x3 array of [x, y, z]
    xyz = np.column_stack((xx.ravel(), yy.ravel(), z_grid.ravel()))

    return xyz


def generate_faces_from_grid(n_row, n_col):
    """
    Generate triangle face indices for a (n_row x n_col) grid of vertices,
    assuming the vertices are flattened row-major (C-style) to a 1D array.

    Returns:
    - faces: (2 * (n_row - 1) * (n_col - 1), 3) int NumPy array of triangle indices
    """
    # Create 2D grid of top-left corner indices for each quad
    i = np.arange(n_row - 1)
    j = np.arange(n_col - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')  # shape: (n_row-1, n_col-1)

    # Flatten grid of cell indices
    ii = ii.ravel()
    jj = jj.ravel()

    # Convert 2D grid indices to flat indices in the 1D vertex array
    top_left = ii + jj * n_col
    bottom_left = ii + (jj + 1) * n_col
    top_right = (ii + 1) + jj * n_col
    bottom_right = (ii + 1) + (jj + 1) * n_col

    # Create two triangles per grid cell
    tri1 = np.stack([top_left, bottom_left, top_right], axis=1)
    tri2 = np.stack([top_right, bottom_left, bottom_right], axis=1)

    # Concatenate both triangles
    faces = np.concatenate([tri1, tri2], axis=0)
    return faces


if __name__ == "__main__":
    terrain = load_bw_image_as_normalized_array("C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation-blender\\data\\terrain_example.png")
    vertices = grid_to_xyz(terrain, start_coordinate=-6, end_coordinate=6).tolist()
    size = terrain.shape[0]
    faces = generate_faces_from_grid(size, size)

    # create mesh in Blender
    perlin_mesh = bpy.data.meshes.new("perlin_mesh")
    perlin_mesh.from_pydata(vertices, [], faces)
    perlin_terrain = bpy.data.objects.new("perlin_terrain", perlin_mesh)
    terrain_collection = bpy.data.collections.new("terrain_collection")
    bpy.context.scene.collection.children.link(terrain_collection)
    terrain_collection.objects.link(perlin_terrain)
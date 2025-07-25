import csv

import imageio.v3 as iio
import numpy as np
import bpy
import os

# DIFFERENT USEFUL KERNEL EXAMPLES
emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

laplacian = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

box_blur_3x3 = np.ones((3, 3), dtype=np.float32)
box_blur_3x3 /= box_blur_3x3.sum()

box_blur_7x7 = np.ones((7, 7), dtype=np.float32)
box_blur_7x7 /= box_blur_7x7.sum()

box_blur_11x11 = np.ones((11, 11), dtype=np.float32)
box_blur_11x11 /= box_blur_11x11.sum()

box_blur_25x25 = np.ones((25, 25), dtype=np.float32)
box_blur_25x25 /= box_blur_25x25.sum()

gaussian_kernel_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
gaussian_kernel_3x3 /= gaussian_kernel_3x3.sum()

gaussian_kernel_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32)
gaussian_kernel_5x5 /= gaussian_kernel_5x5.sum()

kernel_ripple = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

kernel_smoother = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)


def write_vertices_to_csv(vertices, filepath):
    """
    Writes a list of (x, y, z) vertices to a CSV file.

    Parameters:
        vertices: List of tuples/lists like [(x1, y1, z1), (x2, y2, z2), ...]
        filepath: Path to the output CSV file
    """
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for v in vertices:
            if len(v) != 3:
                raise ValueError(f"Invalid vertex (not 3 elements): {v}")
            writer.writerow(v)


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


def apply_convolution(matrix, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)):
    """
        Apply a 2D convolution operation to a matrix using a given square kernel.

        This function performs manual convolution of the input matrix with the specified
        kernel, while preserving the original matrix dimensions. It pads the input using
        edge padding to ensure that border elements are convolved correctly.

        Parameters:
            matrix (np.ndarray): A 2D NumPy array (heightmap or image) to be filtered.
            kernel (np.ndarray, optional): A square 2D NumPy array with odd dimensions
                representing the convolution kernel. Defaults to a 3x3 uniform blur kernel.

        Returns:
            np.ndarray: A 2D NumPy array of the same shape as `matrix`, containing the
            result of the convolution operation.

        Raises:
            AssertionError: If the kernel is not square or its dimensions are not odd.

        Example:
            >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
            >>> kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
            >>> result = apply_convolution(matrix, kernel)
    """

    # Ensure kernel is square and has odd dimensions
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel size must be odd"

    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum() if kernel.sum() != 0 else 1  # normalize if not edge detector

    k = kernel.shape[0] // 2  # padding size

    # Pad the matrix to preserve dimensions after convolution
    padded = np.pad(matrix, pad_width=k, mode='edge')
    output = np.zeros_like(matrix)

    # Perform manual convolution
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            region = padded[i:i + 2 * k + 1, j:j + 2 * k + 1]
            output[i, j] = np.sum(region * kernel)

    return output


if __name__ == "__main__":
    terrain = load_bw_image_as_normalized_array("C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation-blender\\data\\terrain_example.png")
    terrain = apply_convolution(matrix=terrain, kernel=box_blur_11x11)  # smoothen
    vertices = grid_to_xyz(terrain, start_coordinate=-6, end_coordinate=6).tolist()
    size = terrain.shape[0]

    write_vertices_to_csv(vertices=vertices, filepath="C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation\\data\\real_vertices_" + str(size) + ".csv")

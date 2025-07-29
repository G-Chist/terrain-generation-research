"""
PLEASE READ THIS COMMENT BEFORE USING THE CODE!

------------------------------------------
Perlin Noise Terrain Generator for Blender
------------------------------------------

This script generates a procedural terrain mesh using fractal Perlin noise
and imports it directly into Blender. It also optionally applies various
filters (e.g., Gaussian blur) and exports vertex data to a CSV file.

Key Components:
---------------
- Perlin & Fractal Noise Generation:
    Creates smooth 2D noise patterns for terrain modeling.
- Image Filtering:
    Applies convolution kernels (e.g., Gaussian blur, Sobel) to refine features.
- Mesh Conversion:
    Converts the 2D noise map into a 3D mesh grid (x, y, z) suitable for Blender.
- Blender Integration:
    Automatically creates and links a mesh object in the Blender scene.
- CSV Export:
    Saves vertex positions to `vertices.csv` in the current .blend file directory.

Modules Used:
-------------
- `numpy`: Matrix operations and numerical tools.
- `bpy`: Blender's Python API for creating objects and meshes.
- `csv`: Exporting vertex coordinates to CSV.

Usage Instructions:
-------------------
Run the script inside Blender's Scripting workspace.

Author:
-------
Original Perlin/Fractal noise code by Pierre Vigier (MIT License).
Script extended and integrated into Blender by Matvei Shestopalov.

License:
--------
This script includes components under the MIT License. See inline comments for license details.
"""

import numpy as np
import bpy
import csv
from typing import Tuple, List


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


def apply_sobel_magnitude(matrix):
    """
    Applies Sobel X and Y kernels to compute edge magnitude from the input matrix.

    Parameters:
        matrix (np.ndarray): The 2D input array (e.g. terrain heightmap).

    Returns:
        np.ndarray: A matrix of the same shape, with gradient magnitudes.
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gx = apply_convolution(matrix, sobel_x)
    gy = apply_convolution(matrix, sobel_y)

    return np.sqrt(gx ** 2 + gy ** 2)


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


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """

    """MIT License

    Copyright (c) 2019 Pierre Vigier

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE."""

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]] \
               .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """

    """MIT License

        Copyright (c) 2019 Pierre Vigier

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE."""

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def dig_path(
    matrix: np.ndarray,
    kernel: np.ndarray,
    start_cell: Tuple[int, int],
    max_cells: int,
    vert_thresh: float
) -> np.ndarray:
    """
    Digs a path through a grid by applying a kernel to selected cells,
    starting from `start_cell`, and proceeding to neighbors whose height
    difference is within `vert_thresh`, using depth-first traversal.
    Stops after digging into max_cells cells.

    :param matrix: 2D numpy array representing heightmap or terrain
    :param kernel: 2D numpy array (odd-dimensioned) used to modify height values
    :param start_cell: (row, col) tuple of where to start digging
    :param max_cells: maximum number of cells to convolve at
    :param vert_thresh: Maximum allowed height difference to continue path
    :return: Modified matrix (copy) with path dug
    """
    # Get dimensions of the terrain matrix
    h, w = matrix.shape

    # Make a copy of the input matrix to modify without altering the original
    dug_matrix = matrix.copy()

    # Keep track of which cells we've already dug into
    visited = np.zeros_like(matrix, dtype=bool)

    # Stack of cells to visit, initialized with the starting cell
    frontier: List[Tuple[int, int]] = [start_cell]

    # Size and center of the kernel
    kh, kw = kernel.shape
    k_center_h = kh // 2
    k_center_w = kw // 2

    # Counter to limit number of dug cells
    dug_count = 0
    dug_limit = max_cells

    def get_valid_neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns unvisited neighbor cells within vertical threshold.
        Includes 8-connected neighbors (N, S, E, W, NE, NW, SE, SW).
        """
        r, c = cell
        neighbors = []

        # 8 possible directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Ensure neighbor is within bounds and not visited
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                # Only accept neighbors within vertical threshold
                vert_diff = abs(dug_matrix[nr, nc] - dug_matrix[r, c])
                if vert_diff <= vert_thresh:
                    neighbors.append((nr, nc))

        return neighbors

    # Main blurring loop: continue while there are frontier cells and limit not reached
    while frontier and dug_count < dug_limit:
        # Depth-first: pop from end of list (stack behavior)
        r, c = frontier.pop()
        if visited[r, c]:
            continue  # skip already visited cells

        visited[r, c] = True
        dug_count += 1

        # --- Manual convolution: apply kernel over region centered at (r, c) ---
        # Define bounds of region in matrix
        r_start = max(0, r - k_center_h)
        r_end = min(h, r + k_center_h + 1)
        c_start = max(0, c - k_center_w)
        c_end = min(w, c + k_center_w + 1)

        # Extract region
        region = dug_matrix[r_start:r_end, c_start:c_end]
        # Compute blurred region: convolution result
        convolved_region = apply_convolution(region, kernel)

        # Update the matrix region with the blurred values
        dug_matrix[r_start:r_end, c_start:c_end] = convolved_region

        # Add valid neighbors to frontier (DFS order)
        frontier.extend(get_valid_neighbors((r, c)))

    return dug_matrix


def generate_terrain_noise(
        size=1024,                         # size (int): Width and height of the terrain grid.
        res=(8, 8),                        # res (tuple): Base resolution (periods) of the Perlin noise grid.
        octaves=8,                         # octaves (int): Number of fractal noise layers to combine.
        random_seed=123,                   # random_seed (int): Seed for NumPy random generator.
        sea_level=0.5,                     # sea_level (float): Threshold below which terrain is considered 'sea'.
        sky_level=1.0,                     # sky_level (float): Threshold above which terrain is flattened to sky level.
        sea_roughness=0.3,                 # sea_roughness (float): Random fluctuation intensity added near sea level.
        layers=0,                          # layers (int): Number of times to add the terrain to itself.
        trend=None,                        # trend (np.ndarray): An optional trend to add to the surface.
        terrace=False,                     # terrace (bool): Whether or not to apply terracing to the terrain.
        terrace_steepness=11,              # terrace_steepness (int): Defines how steep terracing is.
        terrace_frequency=10,              # terrace_frequency (int): Defines the level of detail for terracing.
        kernels=None                       # kernels (np.ndarray): Optional sequence of convolution kernels to apply.
):
    """
    Generate filtered Perlin-based terrain noise with optional convolution kernels.

    Parameters:
        size (int): Width and height of the terrain grid.
        res (tuple): Base resolution (periods) of the Perlin noise grid.
        octaves (int): Number of fractal noise layers to combine.
        random_seed (int): Seed for NumPy random generator.
        sea_level (float): Threshold below which terrain is considered 'sea'.
        sky_level (float): Threshold above which terrain is flattened to sky level.
        sea_roughness (float): Random fluctuation intensity added near sea level.
        layers (int): Number of times to add the terrain to itself to create a layered terrain.
        trend (np.ndarray): An optional trend to add to the surface.
        kernels (np.ndarray): Optional sequence of convolution kernels to apply.

    Returns:
        np.ndarray: The final filtered terrain heightmap (2D).
    """

    min_amplitude = 0.0
    max_amplitude = 1.0

    shape = (size, size)
    np.random.seed(random_seed)

    # Generate fractal noise
    noise = generate_fractal_noise_2d(shape=shape, res=res, tileable=(True, True), octaves=octaves)

    # Normalize noise
    noise_filtered = np.interp(noise, (noise.min(), noise.max()), (min_amplitude, max_amplitude))
    rand_range = (max_amplitude - min_amplitude) * 0.01

    # Clip to sky level
    noise_filtered = np.where(noise_filtered < sky_level, noise_filtered, sky_level)

    # Add randomness below sea level
    noise_filtered = np.where(
        noise_filtered > sea_level,
        noise_filtered,
        sea_level + np.random.uniform(
            -rand_range * sea_roughness,
            rand_range * sea_roughness,
            noise_filtered.shape
        )
    )

    for _ in range(layers):
        noise_filtered += noise_filtered  # create layered terrain

    # Add trend
    if trend is not None:
        noise_filtered += trend

    # Normalize again
    noise_filtered = np.interp(noise_filtered, (noise_filtered.min(), noise_filtered.max()), (min_amplitude, max_amplitude))

    # Terrace the noise
    if terrace:
        freq = terrace_frequency
        step = np.round(noise_filtered * freq) / freq
        noise_filtered = np.sin((noise_filtered - step) * 2.45) ** terrace_steepness + step

    # Apply convolution kernel / kernels
    if kernels is None:
        kernels = []
    elif not isinstance(kernels, (list, tuple)):
        kernels = [kernels]

    for kernel in kernels:
        noise_filtered = apply_convolution(matrix=noise_filtered, kernel=kernel)

    return noise_filtered


if __name__ == '__main__':

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

    # PARAMETERS
    random_seed = 123

    size = 1024
    res = (8, 8)
    octaves = 8

    sea_level = 0.45
    sky_level = 1
    sea_roughness = 0.5

    layers = 5

    kernels = (box_blur_3x3)

    terrace = False
    terrace_steepness = 11
    terrace_frequency = 4

    # DEFINE TREND
    trend_seed = 42

    x_trend = np.linspace(0, 10, size)
    y_trend = np.linspace(0, 10, size)
    X_trend, Y_trend = np.meshgrid(x_trend, y_trend)
    trend = 1*np.sin(X_trend + 2*Y_trend) + 1.45*np.cos(1.7*X_trend + 0.5*Y_trend)

    # GENERATION
    noise_filtered = generate_terrain_noise(size=size,
                                            res=res,
                                            octaves=octaves,
                                            random_seed=random_seed,
                                            sea_level=sea_level,
                                            sky_level=sky_level,
                                            sea_roughness=sea_roughness,
                                            layers=layers,
                                            trend=trend,
                                            terrace=terrace,
                                            terrace_steepness=terrace_steepness,
                                            terrace_frequency=terrace_frequency,
                                            kernels=kernels)

    # DIG PATHS
    """
    noise_filtered = dig_path(matrix=noise_filtered,
                              kernel=box_blur_11x11,
                              start_cell=(0, 0),
                              max_cells=500,
                              vert_thresh=0.005)

    noise_filtered = dig_path(matrix=noise_filtered,
                              kernel=box_blur_11x11,
                              start_cell=(200, 100),
                              max_cells=500,
                              vert_thresh=0.005)

    noise_filtered = dig_path(matrix=noise_filtered,
                              kernel=box_blur_11x11,
                              start_cell=(570, 300),
                              max_cells=500,
                              vert_thresh=0.005)
    """

    # SMOOTHEN TERRAIN AFTER DIGGING PATH
    # noise_filtered = apply_convolution(matrix=noise_filtered, kernel=box_blur_7x7)

    # Generate mesh data
    vertices = grid_to_xyz(noise_filtered, start_coordinate=-6, end_coordinate=6).tolist()
    faces = generate_faces_from_grid(size, size)

    # Create mesh and object
    perlin_mesh = bpy.data.meshes.new("perlin_mesh")
    perlin_mesh.from_pydata(vertices, [], faces)
    perlin_mesh.update()

    # Add vertex colors
    color_layer = perlin_mesh.vertex_colors.new(name="Col")

    # Normalize z values for color mapping
    z_values = np.array([v[2] for v in vertices])
    z_min, z_max = z_values.min(), z_values.max()


    def lerp(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

    def height_to_color(z):
        norm_z = (z - z_min) / (z_max - z_min)

        # Define key color points (RGBA)
        white = (0.9, 0.9, 0.9, 1.0)
        gray  = (0.3, 0.3, 0.3, 1.0)
        green = (0.196, 0.231, 0.011, 1.0)
        brown = (0.5, 0.37, 0.235, 1.0)

        if norm_z >= 0.4:
            # White to Gray
            t = (norm_z - 0.4) / 0.6
            return lerp(gray, white, t)
        elif norm_z >= 0.1:
            # Gray to Brown
            t = (norm_z - 0.1) / 0.3
            return lerp(brown, gray, t)
        elif norm_z >= 0.0:
            # Brown to Green
            t = norm_z / 0.1
            return lerp(green, brown, t)
        else:
            return green  # fallback for any value below z_min


    # Assign colors to each face's loops
    for poly in perlin_mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = perlin_mesh.loops[loop_idx].vertex_index
            z = vertices[vert_idx][2]
            color_layer.data[loop_idx].color = height_to_color(z)

    # Create object and collection
    perlin_terrain = bpy.data.objects.new("perlin_terrain", perlin_mesh)
    terrain_collection = bpy.data.collections.new("terrain_collection")
    bpy.context.scene.collection.children.link(terrain_collection)
    terrain_collection.objects.link(perlin_terrain)

    # Assign material that uses vertex color
    mat = bpy.data.materials.new(name="TerrainMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add new nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
    vc_node = nodes.new(type='ShaderNodeVertexColor')
    vc_node.layer_name = "Col"

    # Connect nodes
    links.new(vc_node.outputs['Color'], diffuse_node.inputs['Color'])
    links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign material
    perlin_mesh.materials.append(mat)
    perlin_terrain.data.materials.append(mat)

    # export vertices to CSV
    """
    blend_path = bpy.path.abspath("//vertices.csv")
    with open(blend_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        writer.writerows(vertices)
    """

    # VISUALIZE
    """
    import matplotlib.pyplot as plt
    plt.imshow(noise_filtered, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.show()
    """

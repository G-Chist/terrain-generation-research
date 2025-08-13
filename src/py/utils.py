import numpy as np
import csv
from typing import List, Tuple
import os
import imageio.v3 as iio
from scipy.ndimage import generic_filter
import rasterio
from PIL import Image
# from pykrige.ok import OrdinaryKriging  <- uncomment for kriging


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
    # normalize if not edge detector
    kernel /= kernel.sum() if kernel.sum() != 0 else 1

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
        [0,  0,  0],
        [1,  2,  1]
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
            shape, (frequency * res[0], frequency *
                    res[1]), tileable, interpolant
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
        # size (int): Width and height of the terrain grid.
        size=1024,
        # res (tuple): Base resolution (periods) of the Perlin noise grid.
        res=(8, 8),
        # octaves (int): Number of fractal noise layers to combine.
        octaves=8,
        # random_seed (int): Seed for NumPy random generator.
        random_seed=123,
        # sea_level (float): Threshold below which terrain is considered 'sea'.
        sea_level=0.5,
        # sky_level (float): Threshold above which terrain is flattened to sky level.
        sky_level=1.0,
        # sea_roughness (float): Random fluctuation intensity added near sea level.
        sea_roughness=0.3,
        # layers (int): Number of times to add the terrain to itself.
        layers=0,
        # trend (np.ndarray): An optional trend to add to the surface.
        trend=None,
        # terrace (bool): Whether or not to apply terracing to the terrain.
        terrace=False,
        # terrace_steepness (int): Defines how steep terracing is.
        terrace_steepness=11,
        # terrace_frequency (int): Defines the level of detail for terracing.
        terrace_frequency=10,
        # kernels (np.ndarray): Optional sequence of convolution kernels to apply.
        kernels=None
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
    noise = generate_fractal_noise_2d(
        shape=shape, res=res, tileable=(True, True), octaves=octaves)

    # Normalize noise
    noise_filtered = np.interp(
        noise, (noise.min(), noise.max()), (min_amplitude, max_amplitude))
    rand_range = (max_amplitude - min_amplitude) * 0.01

    # Clip to sky level
    noise_filtered = np.where(
        noise_filtered < sky_level, noise_filtered, sky_level)

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
    noise_filtered = np.interp(noise_filtered, (noise_filtered.min(
    ), noise_filtered.max()), (min_amplitude, max_amplitude))

    # Terrace the noise
    if terrace:
        freq = terrace_frequency
        step = np.round(noise_filtered * freq) / freq
        noise_filtered = np.sin((noise_filtered - step)
                                * 2.45) ** terrace_steepness + step

    # Apply convolution kernel / kernels
    if kernels is None:
        kernels = []
    elif not isinstance(kernels, (list, tuple)):
        kernels = [kernels]

    for kernel in kernels:
        noise_filtered = apply_convolution(
            matrix=noise_filtered, kernel=kernel)

    return noise_filtered


def load_vertices_from_csv(filepath):
    """
    Reads the list of (x,y,z) vertices from a CSV file.

    Parameters:
        filepath: Path to the CSV file
    """
    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")

    vertices = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 3:
                continue  # skip invalid rows
            try:
                x, y, z = map(float, row[:3])
                vertices.append((x, y, z))
            except ValueError:
                continue  # skip rows with non-numeric data

    return vertices


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
    """
    Loads a .png image into a NumPy grid.

    Args:
        filepath: the path to a .png file.

    Returns:
        a NumPy grid representing grayscale values in the image.
    """
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


def classify_patch(patch):
    """
    Classify a 3x3 elevation patch into a terrain feature type.

    This function implements a rule-based classification system to identify one
    of ten landform patterns (flat, peak, ridge, shoulder, spur, slope, pit,
    valley, footslope, hollow) based on the relative elevation of the center
    cell compared to its 8 neighbors.

    Parameters:
    -----------
    patch : ndarray
    A flattened 3x3 NumPy array (length 9) representing the local elevation
    window. The center element is at index 4.

    Returns:
    --------
    int
    An integer between 0 and 9 encoding the classified terrain feature:

        - 0 : flat
        - 1 : peak
        - 2 : ridge
        - 3 : shoulder
        - 4 : spur
        - 5 : slope
        - 6 : pit
        - 7 : valley
        - 8 : footslope
        - 9 : hollow
    """
    center = patch[4]
    neighbors = np.delete(patch, 4)  # remove center
    diffs = neighbors - center

    if np.all(diffs == 0):
        return 0  # flat
    elif np.all(diffs < 0):
        return 1  # peak
    elif np.all(diffs > 0):
        return 6  # pit
    elif np.count_nonzero(diffs == 0) >= 6:
        return 5  # slope (simple approx: mostly uniform gradient)
    elif np.count_nonzero(diffs > 0) >= 6:
        return 7  # valley
    elif np.count_nonzero(diffs < 0) >= 6:
        return 2  # ridge
    elif np.count_nonzero((diffs > 0) & (np.abs(diffs) > 0.1)) >= 4:
        return 9  # hollow (low near high)
    elif np.count_nonzero((diffs < 0) & (np.abs(diffs) > 0.1)) >= 4:
        return 4  # spur (high near low)
    elif (np.count_nonzero(diffs < 0) >= 3 and
          np.count_nonzero(diffs > 0) >= 3):
        return 3  # shoulder (center higher but mixed context)
    else:
        return 8  # footslope (mixed but flatter than shoulder)


def feature_map(elevation_array):
    """
    Generate a terrain feature map from a 2D elevation array.

    Applies the `classify_patch` function to each 3x3 neighborhood of the input
    elevation array using a sliding window approach. Each cell in the output
    array represents the terrain classification of the corresponding input
    cell.

    Parameters:
    -----------
    elevation_array : ndarray
        A 2D NumPy array representing elevation values (e.g., a digital
        elevation model, DEM).

    Returns:
    --------
    ndarray
        A 2D NumPy array of the same shape as `elevation_array`, where each
        cell contains an integer from 0 to 9 representing a classified terrain
        feature (see `classify_patch` for encoding).
    """
    return generic_filter(elevation_array, classify_patch, size=3, mode='nearest').astype(np.uint8)


def count_features(feature_map):
    """
    Count the number of terrain features in a feature map.

    Parameters:
    -----------
    feature_map : ndarray
        A 2D NumPy array representing the features of a DEM

    Returns:
    --------
    dict
        A dictionary mapping terrain feature names to their counts.
    """
    counts = np.bincount(feature_map.ravel(), minlength=10).tolist()
    labels = [
        "flat",       # 0
        "peak",       # 1
        "ridge",      # 2
        "shoulder",   # 3
        "spur",       # 4
        "slope",      # 5
        "pit",        # 6
        "valley",     # 7
        "footslope",  # 8
        "hollow"      # 9
    ]

    return dict(zip(labels, counts))


def weierstrass_mandelbrot_3d(x, y, D, G, L, gamma, M, n_max):
    """
    Compute the 3D Weierstrass-Mandelbrot function z(x, y).

    Parameters:
        x, y : 2D np.ndarrays
            Meshgrid arrays of spatial coordinates.
        D : float
            Fractal dimension (typically between 2 and 3).
        G : float
            Amplitude roughness coefficient.
        L : float
            Transverse width of the profile.
        gamma : float
            Frequency scaling factor (typically > 1).
        M : int
            Number of ridges (azimuthal angles).
        n_max : int
            Upper cutoff frequency index.

    Returns:
        z : 2D np.ndarray
            The height field generated by the WM function.
    """
    A = L * (G / L) ** (D - 2) * (np.log(gamma) / M) ** 0.5

    z = np.zeros_like(x)

    for m in range(1, M + 1):
        theta_m = np.arctan2(y, x) - np.pi * m / M
        phi_mn = np.random.uniform(
            0, 2 * np.pi, size=n_max + 1)  # random phase per n

        for n in range(n_max + 1):
            gamma_n = gamma ** n
            r = np.sqrt(x ** 2 + y ** 2)
            term = (
                np.cos(phi_mn[n]) -
                np.cos(
                    2 * np.pi * gamma_n * r / L *
                    np.cos(theta_m) + phi_mn[n]
                )
            )
            z += gamma ** ((D - 3) * n) * term

    z *= A
    return z


def save_array_as_grayscale_png(array: np.ndarray, filepath: str) -> None:
    """
    Save a 2D NumPy array as a grayscale PNG image by linearly scaling its values
    to the [0, 255] range based on the array's own min and max.

    Parameters:
    -----------
    array : np.ndarray
        A 2D NumPy array of float values. The values are scaled based on the min
        and max of the array, so they do not need to be in [0, 1].

    filepath : str
        Path where the PNG image will be saved (should end in .png).

    Notes:
    ------
    This behaves as the inverse of a loader that normalizes an image by
    (value - min) / (max - min).
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    min_val = array.min()
    max_val = array.max()

    if max_val - min_val == 0:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        scaled = ((array - min_val) / (max_val - min_val)
                  * 255).astype(np.uint8)

    image = Image.fromarray(scaled, mode='L')
    image.save(filepath)


def save_array_as_grayscale_png_16bit(array: np.ndarray, filepath: str) -> None:
    """
    Save a 2D NumPy array as a 32-bit grayscale PNG image.

    Parameters:
    -----------
    array : np.ndarray
        A 2D NumPy array of float or integer values. Stored as 16-bit per channel.

    filepath : str
        Path where the PNG image will be saved (should end in .png).
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    min_val = array.min()
    max_val = array.max()

    if max_val - min_val == 0:
        scaled = np.zeros_like(array, dtype=np.uint16)
    else:
        scaled = ((array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

    image = Image.fromarray(scaled, mode='I;16')
    image.save(filepath)


def crop_grid_by_percent(grid, center_x_pct, center_y_pct, size):
    """
    Crops an n x n square from a 2D grid using percent-based center coordinates.

    Parameters:
        grid (np.ndarray): 2D input array.
        center_x_pct (float): X center as a percentage (0 to 1).
        center_y_pct (float): Y center as a percentage (0 to 1).
        size (int): Size of the square to crop.

    Returns:
        np.ndarray: Cropped subgrid of shape (size, size)
    """
    h, w = grid.shape
    center_x = int(center_x_pct * h)
    center_y = int(center_y_pct * w)

    half = size // 2
    x_start = max(center_x - half, 0)
    y_start = max(center_y - half, 0)
    x_end = min(x_start + size, h)
    y_end = min(y_start + size, w)

    # Re-adjust start if crop is too close to border
    x_start = max(x_end - size, 0)
    y_start = max(y_end - size, 0)

    return grid[x_start:x_end, y_start:y_end]


def load_elevation_grid(tif_file):
    """
    Load elevation data from a GeoTIFF file as full-resolution NumPy grids.

    Parameters
    ----------
    tif_file : str
        File path to the input GeoTIFF (.tif) containing elevation data.

    Returns
    -------
    elevation_grid : numpy.ndarray, shape (rows, cols)
        2D array of elevation values (float), with no-data replaced by np.nan.

    x_coords : numpy.ndarray, shape (rows, cols)
        2D array of X spatial coordinates (e.g. longitude or easting) corresponding to each cell.

    y_coords : numpy.ndarray, shape (rows, cols)
        2D array of Y spatial coordinates (e.g. latitude or northing) corresponding to each cell.

    Notes
    -----
    - The function preserves full resolution; no downsampling is performed.
    - The coordinate arrays are computed using the GeoTIFF's affine transform and
      represent the spatial location of each grid cell center.
    """
    with rasterio.open(tif_file) as src:
        # Read first band as float32
        elevation = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            elevation[elevation == nodata] = np.nan  # Replace nodata with NaN

        rows, cols = elevation.shape
        x_pix, y_pix = np.meshgrid(np.arange(cols), np.arange(rows))

        x_coords, y_coords = rasterio.transform.xy(
            src.transform, y_pix, x_pix, offset='center')
        x_coords = np.array(x_coords).reshape(elevation.shape)
        y_coords = np.array(y_coords).reshape(elevation.shape)

    return elevation, x_coords, y_coords


"""
def kriging_interpolate(Z, subdivision=10, variogram_model='exponential'):
    n = Z.shape[0]

    # Coordinates for coarse grid
    x_coarse = np.arange(n)
    y_coarse = np.arange(n)
    Xc, Yc = np.meshgrid(x_coarse, y_coarse)
    coords = np.column_stack((Xc.ravel(), Yc.ravel()))
    values = Z.ravel()

    # Fit Kriging model
    OK = OrdinaryKriging(
        coords[:, 0], coords[:, 1], values,
        variogram_model=variogram_model,
        verbose=False, enable_plotting=False
    )

    # Define finer grid (subdivide each cell)
    fine_n = n * subdivision
    x_fine = np.linspace(0, n - 1, fine_n)
    y_fine = np.linspace(0, n - 1, fine_n)

    # Interpolate with Kriging
    Z_interp, _ = OK.execute("grid", x_fine, y_fine)

    return Z_interp
"""


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


if __name__ == "__main__":  # testing
    import matplotlib.pyplot as plt

    # Generate and display Perlin noise
    noise = generate_perlin_noise_2d(shape=(512, 512), res=(8, 8))
    noise_features = feature_map(noise)

    # Load and process real terrain
    real_terrain = load_bw_image_as_normalized_array(
        r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\terrain_example.png"
    )
    real_terrain_features = feature_map(real_terrain)

    # Load and process eroded terrain
    eroded_terrain = load_bw_image_as_normalized_array(
        r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\erosion_generated_terrain.png"
    )
    eroded_terrain_features = feature_map(eroded_terrain)

    # Generate and process Weierstrass–Mandelbrot terrain
    # Create meshgrid
    x_vals = np.linspace(0, 1024, 1000)
    y_vals = np.linspace(0, 1024, 1000)
    x, y = np.meshgrid(x_vals, y_vals)

    # Parameters
    D = 2.5
    G = 1e-1
    L = 100.0
    gamma = 1.5
    M = 10
    n_max = 20

    wm_noise = weierstrass_mandelbrot_3d(x, y, D, G, L, gamma, M, n_max)
    wm_features = feature_map(wm_noise)

    print(
        f"Feature counts for Perlin Noise:         {count_features(noise_features)}")
    print(
        f"Feature counts for Eroded Terrain:       {count_features(eroded_terrain_features)}")
    print(
        f"Feature counts for Real Terrain:         {count_features(real_terrain_features)}")
    print(
        f"Feature counts for W-M Fractal Terrain:  {count_features(wm_features)}")

    # Set up a 4x2 subplot
    fig, axs = plt.subplots(4, 2, figsize=(12, 13))

    axs[0, 0].imshow(noise, cmap='gray', interpolation='lanczos')
    axs[0, 0].set_title("Perlin Noise")
    axs[0, 0].axis('off')
    fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0])

    axs[0, 1].imshow(noise_features, cmap='gray', interpolation='lanczos')
    axs[0, 1].set_title("Perlin Noise Feature Map")
    axs[0, 1].axis('off')
    fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1])

    axs[1, 0].imshow(real_terrain, cmap='gray', interpolation='lanczos')
    axs[1, 0].set_title("Real Terrain")
    axs[1, 0].axis('off')
    fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0])

    axs[1, 1].imshow(real_terrain_features, cmap='gray',
                     interpolation='lanczos')
    axs[1, 1].set_title("Real Terrain Feature Map")
    axs[1, 1].axis('off')
    fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1])

    axs[2, 0].imshow(eroded_terrain, cmap='gray', interpolation='lanczos')
    axs[2, 0].set_title("Erosion-Generated Terrain")
    axs[2, 0].axis('off')
    fig.colorbar(axs[2, 0].images[0], ax=axs[2, 0])

    axs[2, 1].imshow(eroded_terrain_features,
                     cmap='gray', interpolation='lanczos')
    axs[2, 1].set_title("Erosion-Generated Terrain Feature Map")
    axs[2, 1].axis('off')
    fig.colorbar(axs[2, 1].images[0], ax=axs[2, 1])

    axs[3, 0].imshow(wm_noise, cmap='gray', interpolation='lanczos')
    axs[3, 0].set_title("W–M Fractal Terrain")
    axs[3, 0].axis('off')
    fig.colorbar(axs[3, 0].images[0], ax=axs[3, 0])

    axs[3, 1].imshow(wm_features, cmap='gray', interpolation='lanczos')
    axs[3, 1].set_title("W–M Feature Map")
    axs[3, 1].axis('off')
    fig.colorbar(axs[3, 1].images[0], ax=axs[3, 1])

    plt.tight_layout()
    plt.show()

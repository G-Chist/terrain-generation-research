import numpy as np
import bpy
import csv


def blur_matrix(matrix, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)):
    # Ensure kernel is square and has odd dimensions
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel size must be odd"

    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum()  # Normalize the kernel

    k = kernel.shape[0] // 2  # Padding size

    # Pad the matrix to preserve dimensions after convolution
    padded = np.pad(matrix, pad_width=k, mode='edge')
    output = np.zeros_like(matrix)

    # Perform manual convolution
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            region = padded[i:i + 2 * k + 1, j:j + 2 * k + 1]
            output[i, j] = np.sum(region * kernel)

    return output


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


if __name__ == '__main__':
    # PARAMETERS
    random_seed = 123

    size = 1024
    res = (8, 8)
    octaves = 8

    min_amplitude = 0
    max_amplitude = 1
    sea_level = 0.5
    sky_level = 1
    sea_roughness = 0.3

    # GENERATION
    shape = (size, size)  # does not seem to work for non-1x1 grids
    np.random.seed(random_seed)
    noise = generate_fractal_noise_2d(shape=shape, res=res, octaves=octaves)

    # SCALING, TRANSFORMING, FILTERING
    noise_filtered = np.interp(noise, (noise.min(),
                                       noise.max()),
                               (min_amplitude, max_amplitude))

    rand_range = (max_amplitude - min_amplitude) * 0.01

    noise_filtered = np.where(noise_filtered < sky_level,
                              noise_filtered,
                              sky_level)

    noise_filtered = np.where(noise_filtered > sea_level,
                              noise_filtered,
                              sea_level + np.random.uniform(-rand_range * sea_roughness,
                                                            rand_range * sea_roughness,
                                                            noise_filtered.shape)
                              )

    noise_filtered = blur_matrix(noise_filtered,
                                 kernel=np.ones((41, 41), dtype=np.float32))  # 41x41 ones kernel to blur the noise

    vertices = grid_to_xyz(noise_filtered, start_coordinate=-6, end_coordinate=6).tolist()
    faces = generate_faces_from_grid(shape[0], shape[1])

    # create mesh in Blender
    perlin_mesh = bpy.data.meshes.new("perlin_mesh")
    perlin_mesh.from_pydata(vertices, [], faces)
    perlin_terrain = bpy.data.objects.new("perlin_terrain", perlin_mesh)
    terrain_collection = bpy.data.collections.new("terrain_collection")
    bpy.context.scene.collection.children.link(terrain_collection)
    terrain_collection.objects.link(perlin_terrain)

    # export vertices to CSV
    blend_path = bpy.path.abspath("//vertices.csv")
    with open(blend_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        writer.writerows(vertices)

    # VISUALIZE
    # import matplotlib.pyplot as plt
    # plt.imshow(noise_filtered, cmap='gray', interpolation='lanczos')
    # plt.colorbar()
    # plt.show()

# -------------------------------------------------------------------
# INSTRUCTIONS: How to view generated Perlin noise terrain in Blender
# -------------------------------------------------------------------
# 1. Open Blender and switch to the "Scripting" workspace.
#    - This workspace includes a text editor, console, and 3D viewport.
#
# 2. In the Text Editor panel:
#    - Click "New" to create a new script.
#    - Paste this entire Python script into the editor.
#
# 3. Run the script:
#    - Press Alt + P with your cursor in the text editor,
#      OR click the "Run Script" (▶) button at the top.
#
# 4. View the terrain:
#    - In the 3D viewport, navigate using the mouse to locate the object.
#    - Press 'A' to select all, then 'Home' to frame everything in view.
#    - You should see a mesh named "perlin_terrain" in the scene.
#
# 5. Check the "Outliner" panel:
#    - Look for a collection named "terrain_collection".
#    - Inside it is the object "perlin_terrain" generated by the script.
#
# 6. You can now modify, texture, or export the terrain as needed.
#
# Optional:
#    - Change parameters in the function call to generate different terrain.
#    - Re-run the script each time to update the mesh.
#    - Change random seed

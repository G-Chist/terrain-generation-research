import bpy
import random


random.seed(123)  # change if needed


def interpolate(point_0, point_1, weight):
    # performs linear interpolation between two scalar values
    return point_0 + (point_1 - point_0) * weight


class Vector:
    def __init__(self, x, y):
        # initializes a 2d vector with x and y components
        self.x = x
        self.y = y

    def __mul__(self, other):
        # defines dot product between two vectors
        return self.x * other.x + self.y * other.y

    def __add__(self, other):
        # defines vector addition
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        # defines vector subtraction
        return Vector(self.x - other.x, self.y - other.y)


def randomGradient(grad_range):
    # generates a random vector where both components are within [-grad_range, grad_range]
    return Vector(random.uniform(-grad_range, grad_range),
                  random.uniform(-grad_range, grad_range))


def smoothing(val):
    # smoothing function used in perlin interpolation
    # applies a quintic curve: 6t^5 - 15t^4 + 10t^3
    return 6 * (val ** 5) - 15 * (val ** 4) + 10 * (val ** 3)


def generate_perlin_noise(iterations=3, n_row=10, n_col=10, grad_range=1.0, rand_range=1.0,
                          dec_rate=0.35, size=20.0, sea_level=0.2, std_height=True):
    # initializes height grid with zeros
    heights = [[0] * n_col for _ in range(n_row)]

    for _ in range(iterations):
        # increase resolution by doubling number of rows and columns
        n_row *= 2
        n_col *= 2
        # decay random range and gradient magnitude to reduce detail at higher layers
        rand_range *= dec_rate
        grad_range *= dec_rate

        # create gradient vectors for each grid vertex
        vertex_vectors = [[randomGradient(grad_range) for _ in range(n_col + 1)] for _ in range(n_row + 1)]

        # create block center positions offset by (0.5, 0.5) and jittered slightly
        block_vectors = [[randomGradient(0.5) + Vector(i + 0.5, j + 0.5)
                          for j in range(n_col)] for i in range(n_row)]

        # generate small random scalar noise values
        noise_scalars = [[random.uniform(-rand_range, rand_range)
                          for _ in range(n_col + 1)] for _ in range(n_row + 1)]

        # initialize new height grid
        new_heights = [[0] * n_col for _ in range(n_row)]

        # upscale previous heights to match new resolution by duplicating nearest neighbors
        heights = [[heights[i // 2][j // 2] for j in range(n_col)] for i in range(n_row)]

        # apply smoothing by averaging height values in 3x3 neighborhood
        for i in range(n_row):
            for j in range(n_col):
                temp_val = 0
                n_div = 0
                for k in [-1, 0, 1]:
                    for l in [-1, 0, 1]:
                        x = i + k
                        y = j + l
                        if 0 <= x < n_row and 0 <= y < n_col:
                            temp_val += heights[x][y]
                            n_div += 1
                new_heights[i][j] = temp_val / n_div

        # replace old height grid with smoothed version
        heights = new_heights

        # perlin interpolation: refine heights with gradient noise
        for i in range(n_row):
            for j in range(n_col):
                noise = 0
                for layer in range(iterations):
                    # interpolate values across four surrounding corners using smoothing weights
                    noise += interpolate(
                        interpolate(
                            ((block_vectors[i][j] - Vector(i, j)) * vertex_vectors[i][j]) +
                            noise_scalars[i][j],
                            ((block_vectors[i][j] - Vector(i, j + 1)) * vertex_vectors[i][j + 1]) +
                            noise_scalars[i][j + 1],
                            smoothing((block_vectors[i][j] - Vector(i, j)).y)
                        ),
                        interpolate(
                            ((block_vectors[i][j] - Vector(i + 1, j)) * vertex_vectors[i + 1][j]) +
                            noise_scalars[i + 1][j],
                            ((block_vectors[i][j] - Vector(i + 1, j + 1)) * vertex_vectors[i + 1][j + 1]) +
                            noise_scalars[i + 1][j + 1],
                            smoothing((block_vectors[i][j] - Vector(i, j)).y)
                        ),
                        smoothing((block_vectors[i][j] - Vector(i, j)).x)
                    )
                # add accumulated noise to the heightmap
                heights[i][j] += noise

    if std_height:
        # normalize height values to [0, 1]
        min_height = float('inf')
        max_height = float('-inf')
        for i in range(n_row):
            for j in range(n_col):
                min_height = min(min_height, heights[i][j])
                max_height = max(max_height, heights[i][j])

        # rescale each height value to lie between 0 and 1
        heights = [[(heights[i][j] - min_height) / (max_height - min_height)
                    for j in range(n_col)] for i in range(n_row)]

    # convert height values into 3d vertex positions
    vertices = [(-size / 2 + i / n_row * size,
                 -size / 2 + j / n_col * size,
                 heights[i][j] if heights[i][j] > sea_level else sea_level + random.uniform(-rand_range, rand_range))
                for j in range(n_col) for i in range(n_row)]

    # define triangle faces from 2x2 quads in the grid
    faces = []
    for j in range(n_col - 1):
        for i in range(n_row - 1):
            faces.extend([
                [i + j * n_col, i + (j + 1) * n_col, (i + 1) + j * n_col],
                [(i + 1) + j * n_col, i + (j + 1) * n_col, (i + 1) + (j + 1) * n_col]
            ])

    return vertices, faces


# call the function to generate terrain geometry
vertices, faces = generate_perlin_noise(
    iterations=6,        # number of noise layers (more = higher detail)
    n_row=6,             # initial grid rows
    n_col=6,             # initial grid cols
    grad_range=1,        # initial gradient vector magnitude
    rand_range=1,        # initial random noise amplitude
    size=4.0,            # size of terrain in blender units
    sea_level=0.2        # minimum surface height
)

# create a new mesh and fill it with the generated vertex and face data
perlin_mesh = bpy.data.meshes.new("perlin_mesh")
perlin_mesh.from_pydata(vertices, [], faces)

# create an object to hold the mesh
perlin_terrain = bpy.data.objects.new("perlin_terrain", perlin_mesh)

# create a new collection to organize the terrain in the scene
terrain_collection = bpy.data.collections.new('terrain_collection')

# link the collection and object to the current scene
bpy.context.scene.collection.children.link(terrain_collection)
terrain_collection.objects.link(perlin_terrain)

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

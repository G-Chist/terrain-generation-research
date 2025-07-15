import bpy
import random
import csv
import os
import numpy as np
import time

np.random.seed(123)  # change if needed

start_time = time.time()


def interpolate(p0, p1, w):
    return p0 + (p1 - p0) * w


def smoothing(val):
    return 6 * val ** 5 - 15 * val ** 4 + 10 * val ** 3


def generate_perlin_noise(iterations=3, n_row=10, n_col=10, grad_range=1.0, rand_range=1.0,
                          dec_rate=0.35, size=20.0, sea_level=0.2, sea_roughness=5, std_height=True):
    heights = np.zeros((n_row, n_col))

    for _ in range(iterations):
        n_row *= 2
        n_col *= 2
        rand_range *= dec_rate
        grad_range *= dec_rate

        vertex_vectors = np.random.uniform(-grad_range, grad_range, size=(n_row + 1, n_col + 1, 2))
        noise_scalars = np.random.uniform(-rand_range, rand_range, size=(n_row + 1, n_col + 1))

        block_x, block_y = np.meshgrid(np.arange(n_row), np.arange(n_col), indexing='ij')
        block_offsets = np.random.uniform(-0.5, 0.5, size=(n_row, n_col, 2))
        block_centers = np.stack([block_x + 0.5, block_y + 0.5], axis=-1) + block_offsets

        new_heights = np.zeros((n_row, n_col))
        heights = heights.repeat(2, axis=0).repeat(2, axis=1)

        padded_heights = np.pad(heights, 1, mode='edge')
        for i in range(n_row):
            for j in range(n_col):
                new_heights[i, j] = padded_heights[i:i + 3, j:j + 3].mean()

        heights = new_heights

        for i in range(n_row):
            for j in range(n_col):
                bx, by = block_centers[i, j]
                xi, yi = i, j
                sx = smoothing(bx - xi)
                sy = smoothing(by - yi)

                def grad(ix, iy):
                    diff = np.array([bx - ix, by - iy])
                    return np.dot(diff, vertex_vectors[ix, iy]) + noise_scalars[ix, iy]

                noise = interpolate(
                    interpolate(grad(xi, yi), grad(xi, yi + 1), sy),
                    interpolate(grad(xi + 1, yi), grad(xi + 1, yi + 1), sy),
                    sx
                )

                heights[i, j] += noise

    if std_height:
        min_h, max_h = heights.min(), heights.max()
        heights = (heights - min_h) / (max_h - min_h)

    x = np.linspace(-size / 2, size / 2, n_row)
    y = np.linspace(-size / 2, size / 2, n_col)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    zv = np.where(heights > sea_level, heights, sea_level + np.random.uniform(-rand_range*sea_roughness, rand_range*sea_roughness, heights.shape))
    vertices = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    faces = []
    for j in range(n_col - 1):
        for i in range(n_row - 1):
            idx = lambda x, y: x + y * n_col
            faces.extend([
                [idx(i, j), idx(i, j + 1), idx(i + 1, j)],
                [idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)]
            ])

    return vertices.tolist(), faces


# generate geometry
vertices, faces = generate_perlin_noise(iterations=6, n_row=9, n_col=9, grad_range=1,
                                        rand_range=0.1, size=12.0, sea_level=0.5, sea_roughness=10)

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

# export runtime to CSV
blend_path = bpy.path.abspath("//runtime.csv")
with open(blend_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time elapsed"])
    writer.writerow([str(time.time() - start_time)])


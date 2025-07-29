"""
PLEASE READ THIS COMMENT BEFORE USING THE CODE!

------------------------------------------------------
Terrain Loader for Blender - Digital Elevation Importer
------------------------------------------------------

This script imports a terrain mesh into Blender from a .csv file containing a list
of vertices (x, y, z). Each row of the CSV file represents a single 3D point on a
digital elevation map (DEM). Using these vertices, the script automatically
generates triangular faces to construct the full terrain mesh.

Key Components:
---------------
1. Vertex Loader:
   - Reads a .csv file with at least three comma-separated values per row (x, y, z).
   - Skips invalid or non-numeric rows automatically.

2. Face Generator:
   - Assumes the vertices are structured as a grid in row-major order (C-style).
   - Creates two triangular faces for each quad cell in the grid.
   - Supports arbitrary grid dimensions (e.g., 128x128, 512x512).

3. Mesh Builder:
   - Uses Blender's Python API (`bpy`) to create a new mesh and object.
   - Adds the object to a new "terrain_collection" in the active Blender scene.

Modules Used:
-------------
- `numpy` for efficient grid and index calculations.
- `bpy` for creating and linking Blender meshes and objects.
- `csv` for reading the input vertex data.
- `os` for file validation.

Usage Instructions:
-------------------
1. Ensure the input CSV file is structured with x, y, z columns for each vertex.
   Example row: `-6.0, 3.5, 0.12`
2. Update the `csv_path` variable in the `__main__` section to the correct file path.
3. Run the script inside Blender's Scripting workspace:
   - Open Blender.
   - Go to the Scripting tab.
   - Load this script.
   - Click Run Script.
4. The resulting mesh will appear in the scene in a collection called
   `terrain_collection` under the name `real_terrain`.

Author:
-------
Matvei Shestopalov
"""


import numpy as np
import bpy
import os
import csv


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


def load_vertices_from_csv(filepath):
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


if __name__ == "__main__":
    csv_path = r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\real_vertices_512.csv"
    vertices = load_vertices_from_csv(csv_path)
    faces = generate_faces_from_grid(512, 512)  # size is adjustable

    # Create mesh in Blender
    mesh = bpy.data.meshes.new("real_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("real_terrain", mesh)

    # Add to a new collection
    collection = bpy.data.collections.new("terrain_collection")
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(obj)

import csv

import imageio.v3 as iio
import numpy as np
import bpy
import os

from utils import write_vertices_to_csv, load_bw_image_as_normalized_array, grid_to_xyz, apply_convolution, box_blur_25x25, box_blur_11x11


if __name__ == "__main__":
    terrain = load_bw_image_as_normalized_array("C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation-blender\\data\\terrain_example.png")
    terrain = apply_convolution(matrix=terrain, kernel=box_blur_11x11)  # smoothen
    vertices = grid_to_xyz(terrain, start_coordinate=-6, end_coordinate=6).tolist()
    size = terrain.shape[0]

    write_vertices_to_csv(vertices=vertices, filepath="C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation\\data\\real_vertices_" + str(size) + ".csv")

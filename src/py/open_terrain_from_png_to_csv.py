import csv

import imageio.v3 as iio
import numpy as np
# import bpy
import os

from utils import write_vertices_to_csv, load_bw_image_as_normalized_array, grid_to_xyz, apply_convolution, box_blur_25x25, box_blur_11x11, feature_map


if __name__ == "__main__":
    terrain = load_bw_image_as_normalized_array(r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\GAN_generated_terrain.png")
    if terrain.ndim == 3:
        terrain = terrain[:, :, 0]  # take first channel

    terrain = apply_convolution(matrix=terrain, kernel=box_blur_11x11)  # smoothen

    #"""
    # TESTING
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 10))

    axs[0].imshow(terrain, cmap='gray', interpolation='lanczos')
    axs[0].set_title("Real Terrain")
    axs[0].axis('off')
    fig.colorbar(axs[0].images[0], ax=axs[0])

    axs[1].imshow(feature_map(terrain), cmap='gray', interpolation='lanczos')
    axs[1].set_title("Real Terrain Feature Map")
    axs[1].axis('off')
    fig.colorbar(axs[1].images[0], ax=axs[1])

    plt.tight_layout()
    plt.show()
    
    #"""

    vertices = grid_to_xyz(terrain, start_coordinate=-6, end_coordinate=6).tolist()
    size = terrain.shape[0]

    write_vertices_to_csv(vertices=vertices, filepath=r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\real_vertices_" + str(size) + ".csv")

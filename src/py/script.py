import csv

import imageio.v3 as iio
import numpy as np
# import bpy
import os

from utils import write_vertices_to_csv, load_bw_image_as_normalized_array, grid_to_xyz, apply_convolution, \
    box_blur_25x25, box_blur_11x11, feature_map, box_blur_7x7

if __name__ == "__main__":
    terrain = load_bw_image_as_normalized_array(
        r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\GAN_generated_terrain.png")

    terrain = apply_convolution(matrix=terrain, kernel=box_blur_7x7)  # smoothen

    # """
    # TESTING

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 10))

    axs[0].imshow(terrain, cmap='gray', interpolation='lanczos')
    axs[0].set_title("GAN Terrain")
    axs[0].axis('off')
    fig.colorbar(axs[0].images[0], ax=axs[0])

    axs[1].imshow(feature_map(terrain), cmap='gray', interpolation='lanczos')
    axs[1].set_title("GAN Terrain Feature Map")
    axs[1].axis('off')
    fig.colorbar(axs[1].images[0], ax=axs[1])

    plt.tight_layout()
    plt.show()

    # """

    # crop to minimum dimension
    min_dim = min(terrain.shape)
    terrain = terrain[:min_dim, :min_dim]

    np.save(r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\gan_generated_terrain.npy", terrain)

    # CHECKS
    terrain_loaded = np.load(
        r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\gan_generated_terrain.npy")
    print("Shape:", terrain_loaded.shape)
    print("Min:", terrain_loaded.min(), "Max:", terrain_loaded.max())


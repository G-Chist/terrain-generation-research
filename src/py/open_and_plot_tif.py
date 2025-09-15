import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from utils import load_elevation_grid, crop_grid_by_percent, save_array_as_grayscale_png_16bit, convert_16bit_png_to_jpg


if __name__ == "__main__":

    grid, x_coords, y_coords = load_elevation_grid(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\USGS_1M_17_x54y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif")

    styles = 1
    for i in range(10, 90, 5):
        for j in range(10, 90, 5):

            grid_cropped = crop_grid_by_percent(grid, i/100, j/100, 512)

            elevation = grid_cropped

            save_array_as_grayscale_png_16bit(elevation, r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\styles_png\style" + str(styles) + r".png")
            convert_16bit_png_to_jpg(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\styles_png\style" + str(styles) + r".png",
                               r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\styles_jpg\style" + str(styles) + r".jpg",
                               100)

            styles += 1

    # Plot in 2D

    #"""
    plt.imshow(grid, cmap='gray', interpolation='lanczos')  # height map
    plt.colorbar()
    plt.show()
    #"""

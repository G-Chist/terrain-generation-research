import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from utils import load_elevation_grid, crop_grid_by_percent, save_array_as_grayscale_png_16bit, convert_16bit_png_to_jpg


if __name__ == "__main__":

    grid, x_coords, y_coords = load_elevation_grid(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\output_USGS1m.tif")

    datapoints = 1
    for i in range(0, 100, 1):
        for j in range(0, 100, 1):

            grid_cropped = crop_grid_by_percent(grid, i/100, j/100, 512)

            elevation = grid_cropped

            save_array_as_grayscale_png_16bit(elevation, r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\datapoints_png\datapoint" + str(datapoints) + r".png")

            datapoints += 1

    # Plot in 2D

    """
    plt.imshow(grid, cmap='gray', interpolation='lanczos')  # height map
    plt.colorbar()
    plt.show()
    """

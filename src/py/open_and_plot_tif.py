import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from utils import load_elevation_grid, crop_grid_by_percent, save_array_as_grayscale_png_16bit, convert_16bit_png_to_jpg

if __name__ == "__main__":

    grid, x_coords, y_coords = load_elevation_grid(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\USGS_1M_17_x54y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif")

    grid_cropped = crop_grid_by_percent(grid, 90, 0, 512)

    elevation = grid_cropped

    save_array_as_grayscale_png_16bit(elevation, r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\style.png")
    convert_16bit_png_to_jpg(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\style.png",
                       None,
                       100)

    # Plot in 2D
    plt.imshow(elevation, cmap='gray', interpolation='lanczos')  # height map
    plt.colorbar()
    plt.show()

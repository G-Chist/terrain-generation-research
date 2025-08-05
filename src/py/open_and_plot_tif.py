import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from utils import load_elevation_grid

if __name__ == "__main__":

    elevation, x_coords, y_coords = load_elevation_grid(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\USGS_1M_17_x54y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif")

    # Plot in 2D
    plt.imshow(elevation, cmap='gray', interpolation='lanczos')  # height map
    plt.colorbar()
    plt.show()

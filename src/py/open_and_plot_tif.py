import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from utils import load_elevation_grid

if __name__ == "__main__":

    elevation, x_coords, y_coords = load_elevation_grid(r"C:\Users\79140\Downloads\USGS_1M_17_x54y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif")

    # Plot in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x_coords,
        y_coords,
        elevation,
        cmap='terrain', linewidth=0, antialiased=False
    )

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation')

    plt.show()

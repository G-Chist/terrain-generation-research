import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from utils import generate_perlin_noise_2d, load_bw_image_as_normalized_array, crop_grid_by_percent, kriging_interpolate
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

# Coarse grid (known values)
Z = load_bw_image_as_normalized_array(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\terrain_example.png")
Z = crop_grid_by_percent(Z, 30, 30, 28)

Z_interp = kriging_interpolate(Z=Z, subdivision=10, variogram_model='exponential')
n = Z.shape[0]
fine_n = Z_interp.shape[0]

# Plot 2D and 3D versions
fig = plt.figure(figsize=(18, 10))

# 2D plots
for i, (data, title) in enumerate([
    (Z, f"Original Grid ({n}×{n})"),
    (Z_interp, f"Grid After Kriging ({fine_n}×{fine_n})")
]):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.set_title(title)
    im = ax.imshow(data, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

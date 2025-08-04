import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from utils import generate_perlin_noise_2d, load_bw_image_as_normalized_array, crop_grid_by_percent
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

# Coarse grid (known values)
Z = load_bw_image_as_normalized_array(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\terrain_example.png")
Z = crop_grid_by_percent(Z, 30, 30, 28)
n = Z.shape[0]

# Coordinates for coarse grid
x_coarse = np.arange(n)
y_coarse = np.arange(n)
Xc, Yc = np.meshgrid(x_coarse, y_coarse)
coords = np.column_stack((Xc.ravel(), Yc.ravel()))
values = Z.ravel()

# Fit Kriging model
OK = OrdinaryKriging(
    coords[:, 0], coords[:, 1], values,
    variogram_model='exponential', verbose=False, enable_plotting=False
)

# Define finer grid (subdivide each cell)
k = 10
fine_n = n * k
x_fine = np.linspace(0, n - 1, fine_n)
y_fine = np.linspace(0, n - 1, fine_n)
Xf, Yf = np.meshgrid(x_fine, y_fine)

# Interpolate with Kriging
Z_interp, _ = OK.execute("grid", x_fine, y_fine)

# Subdivide using nearest-neighbor
Z_subdivided = zoom(Z, k, order=0)

# Plot 2D and 3D versions
fig = plt.figure(figsize=(18, 10))

# 2D plots
for i, (data, title) in enumerate([
    (Z, "Original Grid"),
    (Z_subdivided, f"Subdivided Grid ({fine_n}×{fine_n})"),
    (Z_interp, f"Kriged Grid ({fine_n}×{fine_n})")
]):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_title(title)
    im = ax.imshow(data, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)

# 3D plots
for i, (data, X, Y, title) in enumerate([
    (Z, Xc, Yc, "Original Grid (3D)"),
    (Z_subdivided, Xf, Yf, "Subdivided Grid (3D)"),
    (Z_interp, Xf, Yf, "Kriged Grid (3D)")
]):
    ax = fig.add_subplot(2, 3, i + 4, projection="3d")
    ax.plot_surface(X, Y, data, cmap="viridis", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# Grid size
n = 70

# Known data points: (x, y, value)
data = np.array([
    [10, 10, 0.5],
    [40, 10, 1.0],
    [25, 25, 1.5],
    [10, 40, 2.0],
    [40, 40, 2.5],
])

# Grid to interpolate on
grid_x = np.linspace(0, n - 1, n)
grid_y = np.linspace(0, n - 1, n)

# Create Ordinary Kriging object
OK = OrdinaryKriging(
    data[:, 0], data[:, 1], data[:, 2],
    variogram_model="spherical",  # or "linear", "exponential", etc.
    verbose=False,
    enable_plotting=False,
)

# Perform Kriging on the grid
z_interp, ss = OK.execute("grid", grid_x, grid_y)

# Plot result
plt.figure(figsize=(6, 5))
plt.imshow(z_interp, origin='lower', extent=(0, n-1, 0, n-1), cmap='terrain')
plt.colorbar(label="Interpolated Value")
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], edgecolor='black', cmap='terrain', label='Data Points')
plt.title("Ordinary Kriging Interpolation")
plt.legend()
plt.tight_layout()
plt.show()

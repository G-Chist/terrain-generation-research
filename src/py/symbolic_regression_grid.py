import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import python
from utils import generate_perlin_noise_2d, apply_convolution, box_blur_25x25, box_blur_11x11

# Generate 2D input data from noise
np.random.seed(123)
size = n_row = n_col = 32
x = np.linspace(0, 1, n_col)
y = np.linspace(0, 1, n_row)
X_grid, Y_grid = np.meshgrid(x, y)
y_target = generate_perlin_noise_2d(shape=(n_row, n_col), res=(8, 8))
y_target = apply_convolution(matrix=y_target, kernel=box_blur_25x25)

# Reshape inputs for symbolic regression
X = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)  # shape: (n_row*n_col, 2)
y_target = y_target.ravel()  # turn array into 1D

# Fit symbolic regression model
model = PySRRegressor(
    niterations=300,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "sin", "exp", "sqrt", "log", "abs"],
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    verbosity=0,
)
model.fit(X, y_target)


# Print all discovered equations
print("\nBest Discovered Equation:")
print(model.get_best())

# Convert symbolic expression to valid Python syntax

sym_expr = model.get_best()["sympy_format"]
print("\nPython Syntax Formula:")
print(python(sym_expr))

# Plotting: ground truth as dots, fit as surface
x_grid = np.linspace(0, 1, 100)
y_grid = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x_grid, y_grid)
grid = np.column_stack([xx.ravel(), yy.ravel()])

# Predict and reshape back to grid
Z_pred = model.predict(grid).reshape(xx.shape)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Ground truth dots
ax.scatter(X[:, 0], X[:, 1], y_target, color='red', alpha=0.5, label="Ground Truth (Perlin Noise)")

# Fitted surface
ax.plot_surface(xx, yy, Z_pred, cmap='viridis', alpha=0.8)

ax.set_title("Symbolic Regression Fit vs Perlin Noise")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.legend()
plt.tight_layout()
plt.show()


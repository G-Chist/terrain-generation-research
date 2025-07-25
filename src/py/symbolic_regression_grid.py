import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import python

# Generate random 2D input data
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(200, 2))  # 200 samples, 2 features


# Define target function with noise
def f(x, y):
    return np.sin(x) + np.cos(y)


y = f(X[:, 0], X[:, 1]) + np.random.normal(0, 0.1, size=200)

# Symbolic Regression
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp"],
    maxsize=20,
    model_selection="best",
    verbosity=1,
)

model.fit(X, y)

# Print all discovered equations
print("\nBest Discovered Equation:")
print(model.get_best())

# Convert symbolic expression to valid Python syntax

sym_expr = model.get_best()["sympy_format"]
print("\nPython Syntax Formula:")
print(python(sym_expr))

# Plotting: ground truth as dots, fit as surface
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
xx, yy = np.meshgrid(x_grid, y_grid)
grid = np.column_stack([xx.ravel(), yy.ravel()])

zz_pred = model.predict(grid).reshape(xx.shape)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Ground truth dots
ax.scatter(X[:, 0], X[:, 1], y, color='red', alpha=0.5, label="Ground Truth (noisy)")

# Fitted surface
ax.plot_surface(xx, yy, zz_pred, cmap='viridis', alpha=0.8, label="SR Fit")

ax.set_title("Symbolic Regression Fit vs Ground Truth")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.legend()
plt.tight_layout()
plt.show()

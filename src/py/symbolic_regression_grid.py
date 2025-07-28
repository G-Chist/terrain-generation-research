import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import python


def apply_convolution(matrix, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)):
    """
        Apply a 2D convolution operation to a matrix using a given square kernel.

        This function performs manual convolution of the input matrix with the specified
        kernel, while preserving the original matrix dimensions. It pads the input using
        edge padding to ensure that border elements are convolved correctly.

        Parameters:
            matrix (np.ndarray): A 2D NumPy array (heightmap or image) to be filtered.
            kernel (np.ndarray, optional): A square 2D NumPy array with odd dimensions
                representing the convolution kernel. Defaults to a 3x3 uniform blur kernel.

        Returns:
            np.ndarray: A 2D NumPy array of the same shape as `matrix`, containing the
            result of the convolution operation.

        Raises:
            AssertionError: If the kernel is not square or its dimensions are not odd.

        Example:
            >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
            >>> kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
            >>> result = apply_convolution(matrix, kernel)
    """

    # Ensure kernel is square and has odd dimensions
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel size must be odd"

    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum() if kernel.sum() != 0 else 1  # normalize if not edge detector

    k = kernel.shape[0] // 2  # padding size

    # Pad the matrix to preserve dimensions after convolution
    padded = np.pad(matrix, pad_width=k, mode='edge')
    output = np.zeros_like(matrix)

    # Perform manual convolution
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            region = padded[i:i + 2 * k + 1, j:j + 2 * k + 1]
            output[i, j] = np.sum(region * kernel)

    return output


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """

    """MIT License

    Copyright (c) 2019 Pierre Vigier

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE."""

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]] \
               .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """

    """MIT License

        Copyright (c) 2019 Pierre Vigier

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE."""

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


# Kernels to blur noise
box_blur_3x3 = np.ones((3, 3), dtype=np.float32)
box_blur_3x3 /= box_blur_3x3.sum()

box_blur_7x7 = np.ones((7, 7), dtype=np.float32)
box_blur_7x7 /= box_blur_7x7.sum()

box_blur_11x11 = np.ones((11, 11), dtype=np.float32)
box_blur_11x11 /= box_blur_11x11.sum()

box_blur_25x25 = np.ones((25, 25), dtype=np.float32)
box_blur_25x25 /= box_blur_25x25.sum()

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
    niterations=250,
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


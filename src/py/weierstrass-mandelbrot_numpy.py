import numpy as np
from utils import weierstrass_mandelbrot_3d


if __name__ == '__main__':  # example
    import matplotlib.pyplot as plt

    # Create meshgrid
    x_vals = np.linspace(0, 1024, 1000)
    y_vals = np.linspace(0, 1024, 1000)
    x, y = np.meshgrid(x_vals, y_vals)

    # Parameters
    D = 2.5
    G = 1e-1
    L = 100.0
    gamma = 1.5
    M = 10
    n_max = 20

    # Compute WM surface
    z = weierstrass_mandelbrot_3d(x, y, D, G, L, gamma, M, n_max)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', linewidth=0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('3D Weierstrass-Mandelbrot Surface')
    plt.show()

    plt.imshow(z, cmap='gray', interpolation='lanczos')  # height map
    plt.colorbar()
    plt.show()

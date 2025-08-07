import torch
import numpy as np


def weierstrass_mandelbrot_3d_torch(x, y, D, G, L, gamma, M, n_max, device='cuda'):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    A = L * (G / L) ** (D - 2) * (torch.log(torch.tensor(gamma)) / M) ** 0.5

    z = torch.zeros_like(x)

    r = torch.sqrt(x ** 2 + y ** 2)
    for m in range(1, M + 1):
        theta_m = torch.atan2(y, x) - torch.pi * m / M
        phi_mn = torch.rand(n_max + 1, device=device) * 2 * torch.pi

        for n in range(n_max + 1):
            gamma_n = gamma ** n
            term = (
                torch.cos(phi_mn[n]) -
                torch.cos(2 * torch.pi * gamma_n * r / L * torch.cos(theta_m) + phi_mn[n])
            )
            z += gamma ** ((D - 3) * n) * term

    return (A * z).cpu().numpy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    size = 400
    res = 2000
    random_seed = 123

    np.random.seed(random_seed)

    # Create meshgrid
    x_vals = np.linspace(0, size, res)
    y_vals = np.linspace(0, size, res)
    x, y = np.meshgrid(x_vals, y_vals)

    # Example parameters
    D = 2.5
    G = 1e-1
    L = 100.0
    gamma = 1.5
    M = 10
    n_max = 10

    # Compute WM surfaces
    z1 = weierstrass_mandelbrot_3d_torch(
        x=x, y=y, D=2.2, G=1e-6, L=L, gamma=gamma, M=16, n_max=n_max)
    z2 = weierstrass_mandelbrot_3d_torch(
        x=x, y=y, D=2.45, G=8e-8, L=L, gamma=gamma, M=32, n_max=n_max)
    z3 = weierstrass_mandelbrot_3d_torch(
        x=x, y=y, D=2.45, G=1e-8, L=L, gamma=gamma, M=64, n_max=n_max)

    # Compute Hadamard product
    z = z1 * z2 * z3

    # Normalize to [0,1]
    z = np.interp(z, (z.min(), z.max()), (0, 1))

    plt.imshow(z, cmap='gray', interpolation='lanczos')  # height map
    plt.colorbar()
    plt.show()

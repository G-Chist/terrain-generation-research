import torch
import math


def default_interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d_torch(shape, res, tileable=(False, False), interpolant=default_interpolant, device='cpu'):
    """
    Generate 2D Perlin noise using PyTorch.

    Args:
        shape: Tuple[int, int], output shape (must be multiple of res).
        res: Tuple[int, int], number of periods (must divide shape).
        tileable: Tuple[bool, bool], whether noise is tileable in each axis.
        interpolant: Callable, interpolation function.
        device: 'cpu' or 'cuda'.

    Returns:
        torch.Tensor of shape `shape`, with values in [-1, 1].
    """
    if shape[0] % res[0] != 0 or shape[1] % res[1] != 0:
        raise ValueError("Shape must be multiple of res")

    d = (shape[0] // res[0], shape[1] // res[1])
    delta = (res[0] / shape[0], res[1] / shape[1])

    # Grid of coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, res[0], steps=shape[0], device=device, dtype=torch.float32, requires_grad=False) % 1,
        torch.linspace(0, res[1], steps=shape[1], device=device, dtype=torch.float32, requires_grad=False) % 1,
        indexing='xy'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)

    # Generate random gradients
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]

    gradients = gradients.repeat_interleave(d[0], dim=0).repeat_interleave(d[1], dim=1)

    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]

    # Compute dot products (ramps)
    dot = lambda grad, offset: (grid + offset).mul(grad).sum(dim=2)

    n00 = dot(g00, torch.zeros_like(grid))
    n10 = dot(g10, torch.tensor([-1, 0], device=device))
    n01 = dot(g01, torch.tensor([0, -1], device=device))
    n11 = dot(g11, torch.tensor([-1, -1], device=device))

    # Interpolate
    t = interpolant(grid)
    t_x = t[..., 0]
    t_y = t[..., 1]

    n0 = n00 * (1 - t_x) + t_x * n10
    n1 = n01 * (1 - t_x) + t_x * n11
    return torch.sqrt(torch.tensor(2.0, device=device)) * ((1 - t_y) * n0 + t_y * n1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    noise = generate_perlin_noise_2d_torch(
        shape=(2000, 2000),
        res=(8, 8),
        tileable=(True, True),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    plt.imshow(noise.cpu(), cmap='gray')
    plt.colorbar()
    plt.title("Perlin Noise (Torch Accelerated)")
    plt.show()

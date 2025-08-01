from utils import generate_fractal_noise_2d, save_array_as_grayscale_png, generate_perlin_noise_2d

if __name__ == "__main__":
    perlin = generate_perlin_noise_2d(shape=(512, 512), res=(8, 8))
    save_array_as_grayscale_png(perlin, r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\perlin_png.png")

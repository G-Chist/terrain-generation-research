from utils import generate_fractal_noise_2d, save_array_as_grayscale_png, generate_perlin_noise_2d, weierstrass_mandelbrot_3d

if __name__ == "__main__":  # TODO: EXPAND DATASET + CSV TO INCLUDE W-M NOISE WITH VARYING PARAMS AND NON-FRACTAL PERLIN NOISE!!!
    perlin = generate_perlin_noise_2d(shape=(512, 512), res=(8, 8))
    save_array_as_grayscale_png(perlin, r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\perlin_png.png")

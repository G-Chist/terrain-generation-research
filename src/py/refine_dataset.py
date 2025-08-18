import os
from PIL import Image
import numpy as np

IMAGE_SIZE = 128
WHITE_THRESHOLD = 200
BLACK_THRESHOLD = 50
MAX_FRACTION = 0.1
MAX_FRACTION_WHITE = 0.9
STEP = IMAGE_SIZE // 4  # sliding window step

def generate_cropped_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir) if f.endswith("_h.png")]

    for fname in img_files:
        img_path = os.path.join(input_dir, fname)
        image = Image.open(img_path).convert("L")
        img_np = np.array(image)
        height, width = img_np.shape

        crop_count = 0
        # Sliding window to find all valid crops
        for top in range(0, height - IMAGE_SIZE + 1, STEP):
            for left in range(0, width - IMAGE_SIZE + 1, STEP):
                crop = img_np[top:top+IMAGE_SIZE, left:left+IMAGE_SIZE]
                white_fraction = np.mean(crop > WHITE_THRESHOLD)
                black_fraction = np.mean(crop < BLACK_THRESHOLD)

                if black_fraction < MAX_FRACTION and white_fraction < MAX_FRACTION_WHITE:
                    crop_img = Image.fromarray(crop)
                    # unique filename for each crop
                    base_name = os.path.splitext(fname)[0]
                    save_path = os.path.join(output_dir, f"{base_name}_crop{crop_count}_h.png")
                    crop_img.save(save_path)
                    crop_count += 1

        if crop_count == 0:
            print(f"No valid crops found in {fname}")

# Example usage
input_dir = r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\archive\_dataset"
output_dir = r"C:\Users\mshestopalov\PycharmProjects\procedural-terrain-generation\data\processed_dataset_less_water"
generate_cropped_dataset(input_dir, output_dir)

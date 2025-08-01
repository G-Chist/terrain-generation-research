import os
import csv
from pathlib import Path
from utils import load_bw_image_as_normalized_array, feature_map, count_features

# Input folders
real_dir = Path(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\_dataset")
eroded_dir = Path(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\dataset_erosion_perlin\dataset")

# Output CSV file
output_csv = r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\terrain_features_dataset.csv"

# Feature labels in order
feature_labels = [
    "flat", "peak", "ridge", "shoulder", "spur",
    "slope", "pit", "valley", "footslope", "hollow"
]

# CSV writing
with open(output_csv, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=feature_labels + ["realness"])
    writer.writeheader()

    # ---- Real heightmaps (realness = 1.0, only first 300) ----
    real_images = list(real_dir.glob("*_h.png"))[:300]
    for file_path in real_images:
        elevation = load_bw_image_as_normalized_array(str(file_path))
        features = feature_map(elevation)
        counts = count_features(features)

        total = sum(counts.values())
        proportions = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
        proportions["realness"] = 1.0

        writer.writerow(proportions)
        print(f"Real terrain features added!")

    # ---- Eroded (realness = 0.5) and Perlin (realness = 0.1) ----
    for file_path in eroded_dir.glob("*.png"):
        filename = file_path.name
        if filename.endswith("a.png"):
            score = 0.5
        elif filename.endswith("b.png"):
            score = 0.1
        else:
            continue  # ignore unrelated files

        elevation = load_bw_image_as_normalized_array(str(file_path))
        features = feature_map(elevation)
        counts = count_features(features)

        total = sum(counts.values())
        proportions = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
        proportions["realness"] = score

        writer.writerow(proportions)
        print(f"Fake terrain features added!")

print(f"Dataset CSV created: {output_csv}")

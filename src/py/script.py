import os
import csv
from pathlib import Path
from utils import load_bw_image_as_normalized_array, feature_map, count_features

# Input folders
real_dir = Path(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\_dataset")

# Output CSV file
output_csv = r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\real_terrain_features_dataset.csv"

# Feature labels in order
feature_labels = [
    "flat", "peak", "ridge", "shoulder", "spur",
    "slope", "pit", "valley", "footslope", "hollow"
]

# CSV writing
with open(output_csv, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=feature_labels + ["realness"])
    writer.writeheader()

    # ---- Real heightmaps (realness = 1.0) ----
    real_images = list(real_dir.glob("*_h.png"))
    for file_path in real_images:
        elevation = load_bw_image_as_normalized_array(str(file_path))
        features = feature_map(elevation)
        counts = count_features(features)

        total = sum(counts.values())
        proportions = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
        proportions["realness"] = 1.0

        writer.writerow(proportions)
        print(f"Real terrain features added!")


print(f"Dataset CSV created: {output_csv}")

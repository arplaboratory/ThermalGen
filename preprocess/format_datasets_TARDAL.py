import os
import webdataset as wds
from tqdm import tqdm
import cv2
import json
import numpy as np
import random

# Dataset Paths
RAW_DATA_PATH_TARDAL = "datasets_raw/TARDAL"
OUTPUT_PATH_TARDAL = "datasets_preprocess/TARDAL"

# Create output directory
os.makedirs(OUTPUT_PATH_TARDAL, exist_ok=True)

def pair_samples(split_folder, color_folder, thermal_folder):
    """
    Returns a list of (color_file, thermal_file) that exist in both subfolders.
    """
    color_folder_path = os.path.join(RAW_DATA_PATH_TARDAL, split_folder, color_folder)
    thermal_folder_path = os.path.join(RAW_DATA_PATH_TARDAL, split_folder, thermal_folder)

    color_files = sorted(os.listdir(color_folder_path))
    thermal_files = set(os.listdir(thermal_folder_path))

    paired = []
    for cfile in color_files:
        if cfile in thermal_files:
            paired.append((cfile, cfile))
    return paired

def write_shards(paired_samples, split_name, split_output_path, color_subfolder, thermal_subfolder):
    """
    Writes out shards for the given sample list (paired_samples) to 'split_output_path'.
    `split_name` is just used to display which split we are writing (train, val, etc.).
    """
    os.makedirs(split_output_path, exist_ok=True)

    num_samples = len(paired_samples)
    print(f"\n[{split_name.upper()}] Number of samples: {num_samples}")

    shard_size = 1000
    num_shards = (num_samples + shard_size - 1) // shard_size

    with tqdm(total=num_samples, desc=f"Writing {split_name} shards") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(split_output_path, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for sample_idx in range(shard_idx * shard_size, 
                                        min((shard_idx + 1) * shard_size, num_samples)):
                    color_file, thermal_file = paired_samples[sample_idx]
                    sample_key = color_file.split(".")[0]
                    try:
                        # Construct full file paths
                        color_path = os.path.join(RAW_DATA_PATH_TARDAL, split_name, color_subfolder, color_file)
                        thermal_path = os.path.join(RAW_DATA_PATH_TARDAL, split_name, thermal_subfolder, thermal_file)

                        # Read IR image
                        ir_img = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
                        if ir_img is None:
                            raise ValueError(f"Failed to read IR image: {thermal_path}")
                        _, ir_encoded = cv2.imencode('.png', ir_img)

                        # Read RGB image
                        rgb_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
                        if rgb_img is None:
                            raise ValueError(f"Failed to read RGB image: {color_path}")
                        _, rgb_encoded = cv2.imencode('.png', rgb_img)

                        # Write to WebDataset
                        sink.write({
                            "__key__": sample_key,
                            "thermal.png": ir_encoded.tobytes(),
                            "color.png": rgb_encoded.tobytes(),
                        })
                    except Exception as e:
                        print(f"Error processing sample {sample_key}: {e}")
                    pbar.update(1)

    # Write a metadata JSON file with number of samples
    json_path = os.path.join(split_output_path, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": num_samples}, f)
    print(f"Metadata saved in {json_path}")

# ─────────────────────────────────────────────────────────
# 1) Process M3FD_Detection (TARDAL): 80% train / 20% val
# ─────────────────────────────────────────────────────────
split_folder = "M3FD_Detection"
color_folder_name = "Vis"
thermal_folder_name = "Ir"

all_pairs = pair_samples(split_folder, color_folder_name, thermal_folder_name)
print(f"Total paired samples in {split_folder}: {len(all_pairs)}")

# Shuffle to randomize before splitting
random.shuffle(all_pairs)

# 80:20 split
train_size = int(0.8 * len(all_pairs))
train_pairs = all_pairs[:train_size]
test_pairs = all_pairs[train_size:]

train_output = os.path.join(OUTPUT_PATH_TARDAL, split_folder, "train")
test_output = os.path.join(OUTPUT_PATH_TARDAL, split_folder, "test")

# Write train and val shards
write_shards(train_pairs, split_folder, train_output, color_folder_name, thermal_folder_name)
write_shards(test_pairs, split_folder, test_output, color_folder_name, thermal_folder_name)

# ───────────────────────────────────────────────────
# 2) roadscene: 100% val
# ───────────────────────────────────────────────────
split_folder = "roadscene"
color_folder_name = "vi"
thermal_folder_name = "ir"

roadscene_pairs = pair_samples(split_folder, color_folder_name, thermal_folder_name)
print(f"Total paired samples in {split_folder}: {len(roadscene_pairs)}")

val_output = os.path.join(OUTPUT_PATH_TARDAL, split_folder, "val")

# All as val
write_shards(roadscene_pairs, split_folder, val_output, color_folder_name, thermal_folder_name)

# ───────────────────────────────────────────────────
# 3) tno: 100% val
# ───────────────────────────────────────────────────
split_folder = "tno"
color_folder_name = "vi"
thermal_folder_name = "ir"

tno_pairs = pair_samples(split_folder, color_folder_name, thermal_folder_name)
print(f"Total paired samples in {split_folder}: {len(tno_pairs)}")

val_output = os.path.join(OUTPUT_PATH_TARDAL, split_folder, "val")

# All as val
write_shards(tno_pairs, split_folder, val_output, color_folder_name, thermal_folder_name)

print("Dataset formatting completed.")

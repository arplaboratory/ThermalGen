import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import io
import cv2
import json
import numpy as np

# Dataset Paths
RAW_DATA_PATH_AVIID = "datasets_raw/AVIID/AVIID"
OUTPUT_PATH_AVIID = "datasets_preprocess/AVIID"

# Create output directories
os.makedirs(OUTPUT_PATH_AVIID, exist_ok=True)

# Function to format Caltech dataset
def process_AVIID(split):

    split_output_path = os.path.join(OUTPUT_PATH_AVIID, split)

    color_files = sorted(os.listdir(os.path.join(RAW_DATA_PATH_AVIID,split+"A")))
    thermal_files = set(os.listdir(os.path.join(RAW_DATA_PATH_AVIID,split+"B")))

    paired_samples = [] #list(zip(color_files, thermal_files))
    for color_file in color_files:
        if color_file in thermal_files:
            paired_samples.append((color_file, color_file))

    print(len(paired_samples))

    shard_size = 1000
    num_samples = len(paired_samples)
    num_shards = (num_samples + shard_size - 1) // shard_size
    split_output_path = os.path.join(OUTPUT_PATH_AVIID, split)
    os.makedirs(split_output_path, exist_ok=True)

    with tqdm(total=num_samples, desc="Formatting AVIID dataset") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(split_output_path, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for sample_idx in range(shard_idx * shard_size, min((shard_idx + 1) * shard_size, num_samples)):
                    color_file, thermal_file = paired_samples[sample_idx]
                    sample_key = color_file.split(".")[0]
                    try:
                        ir_path = os.path.join(RAW_DATA_PATH_AVIID, split+"B", thermal_file)
                        rgb_path = os.path.join(RAW_DATA_PATH_AVIID, split+"A", color_file)
                        # Read IR image with OpenCV
                        ir_img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
                        if ir_img is None:
                            raise ValueError(f"Failed to read IR image: {ir_path}")
                        # Re-encode as PNG
                        _, ir_image_encoded = cv2.imencode('.png', ir_img)
                        # Read RGB image with OpenCV
                        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                        if rgb_img is None:
                            raise ValueError(f"Failed to read RGB image: {rgb_path}")
                        # Re-encode as PNG
                        _, rgb_image_encoded = cv2.imencode('.png', rgb_img)
                        # Write to WebDataset .tar file
                        sink.write({
                            "__key__": sample_key,        # Sample ID as key
                            "thermal.png": ir_image_encoded.tobytes(),
                            "color.png": rgb_image_encoded.tobytes(),
                        })
                    except Exception as e:
                        print(f"Error processing sample {sample_key}: {e}")
                    pbar.update(1)
    print(f"Total samples formatted for AVIID: {num_samples}")

    # Save the number of samples in a JSON file
    json_path = os.path.join(split_output_path, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": num_samples}, f)
    print(f"Metadata saved in {json_path}")
    
# Process both datasets
process_AVIID("test")
process_AVIID("train")

print("Dataset formatting completed.")
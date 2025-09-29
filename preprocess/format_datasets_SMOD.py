import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import io
import cv2
import json
import glob

# Dataset Paths
RAW_DATA_PATH_SMOD = "datasets_raw/SMOD"
OUTPUT_PATH_SMOD = "datasets_preprocess/SMOD"

# Create output directories
os.makedirs(OUTPUT_PATH_SMOD, exist_ok=True)

# Function to format SMOD dataset
def process_SMOD(folder_name):
    os.makedirs(os.path.join(OUTPUT_PATH_SMOD, folder_name), exist_ok=True)
    
    color_files = sorted(glob.glob(os.path.join(RAW_DATA_PATH_SMOD, folder_name, "*_rgb.jpg")))
    thermal_files = sorted(glob.glob(os.path.join(RAW_DATA_PATH_SMOD, folder_name, "*_tir.jpg")))

    paired_samples = []
    for color_file in color_files:
        thermal_file = color_file.replace("rgb", "tir")  # Adjust this if naming differs
        if thermal_file in thermal_files:
            paired_samples.append((color_file, thermal_file))

    shard_size = 1000
    num_samples = len(paired_samples)
    num_shards = (num_samples + shard_size - 1) // shard_size

    with tqdm(total=num_samples, desc="Formatting SMOD dataset") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(OUTPUT_PATH_SMOD, folder_name, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for sample_idx in range(shard_idx * shard_size, min((shard_idx + 1) * shard_size, num_samples)):
                    color_file, thermal_file = paired_samples[sample_idx]
                    sample_key = color_file.split("/")[-1].split(".")[0]
                    try:
                        ir_path = os.path.join(thermal_file)
                        rgb_path = os.path.join(color_file)
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
    print(f"Total samples formatted for SMOD: {num_samples}")

    # Save the number of samples in a JSON file
    json_path = os.path.join(OUTPUT_PATH_SMOD, folder_name, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": num_samples}, f)
    print(f"Metadata saved in {json_path}")
    
# Process both datasets
process_SMOD("day")
process_SMOD("night")

print("Dataset formatting completed.")

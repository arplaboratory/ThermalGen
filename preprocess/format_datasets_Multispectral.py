import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import io
import cv2
import json
import numpy as np

# Dataset Paths
RAW_DATA_PATH_MIR = "datasets_raw/Multispectral/ir_det_dataset/ConvertedImages"
OUTPUT_PATH_MIR = "datasets_preprocess/Multispectral"

# Create output directories
os.makedirs(OUTPUT_PATH_MIR, exist_ok=True)

def get_non_black_bbox(image):
    """
    Get bounding box of non-black region in the image.
    
    Args:
        image (np.ndarray): Input image (grayscale or color).
        threshold (int): Intensity threshold to consider as non-black.

    Returns:
        (x_min, y_min, x_max, y_max): Bounding box coordinates.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mask = gray > 0
    coords = np.argwhere(mask)

    if coords.size == 0:
        return (0, 0, image.shape[1], image.shape[0])  # Full image if completely black

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return (x_min, y_min, x_max, y_max)

# Function to format Caltech dataset
def process_mir(split):
    COLOR_FOLDER_NAME = "rgb"

    split_output_path = os.path.join(OUTPUT_PATH_MIR, split)

    color_files = sorted(os.listdir(os.path.join(RAW_DATA_PATH_MIR, COLOR_FOLDER_NAME)))
    thermal_files = set(os.listdir(os.path.join(RAW_DATA_PATH_MIR, split)))

    paired_samples = [] #list(zip(color_files, thermal_files))
    for color_file in color_files:
        if color_file in thermal_files:
            paired_samples.append((color_file, color_file))

    shard_size = 1000
    num_samples = len(paired_samples)
    num_shards = (num_samples + shard_size - 1) // shard_size

    with tqdm(total=num_samples, desc="Formatting MIR dataset") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(split_output_path, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for sample_idx in range(shard_idx * shard_size, min((shard_idx + 1) * shard_size, num_samples)):
                    color_file, thermal_file = paired_samples[sample_idx]
                    sample_key = color_file.split(".")[0]
                    try:
                        ir_path = os.path.join(RAW_DATA_PATH_MIR, split, thermal_file)
                        rgb_path = os.path.join(RAW_DATA_PATH_MIR, COLOR_FOLDER_NAME, color_file)
                        # Read IR image with OpenCV
                        ir_img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
                        if ir_img is None:
                            raise ValueError(f"Failed to read IR image: {ir_path}")
                        # Read RGB image with OpenCV
                        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                        assert ir_img.shape[:2] == rgb_img.shape[:2]
                        x_min, y_min, x_max, y_max = get_non_black_bbox(ir_img)
                        cropped = ir_img[y_min:y_max, x_min:x_max]
                        rgb_cropped = rgb_img[y_min:y_max, x_min:x_max]
                        if rgb_img is None:
                            raise ValueError(f"Failed to read RGB image: {rgb_path}")
                         # Re-encode as PNG
                        _, ir_image_encoded = cv2.imencode('.png', cropped)
                        # Re-encode as PNG
                        _, rgb_image_encoded = cv2.imencode('.png', rgb_cropped)
                        # Write to WebDataset .tar file
                        sink.write({
                            "__key__": sample_key,        # Sample ID as key
                            "thermal.png": ir_image_encoded.tobytes(),
                            "color.png": rgb_image_encoded.tobytes(),
                        })
                    except Exception as e:
                        print(f"Error processing sample {sample_key}: {e}")
                    pbar.update(1)
    print(f"Total samples formatted for Multispectral: {num_samples}")

    # Save the number of samples in a JSON file
    json_path = os.path.join(split_output_path, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": num_samples}, f)
    print(f"Metadata saved in {json_path}")
    
# Process both datasets
process_mir("fir")
process_mir("mir")
process_mir("nir")

print("Dataset formatting completed.")

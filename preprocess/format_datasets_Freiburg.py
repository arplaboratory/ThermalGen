import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import io
import cv2
import json
import numpy as np

# Dataset Paths
RAW_DATA_PATH_FREIBURG = "datasets_raw/Freiburg"
OUTPUT_PATH_FREIBURG = "datasets_preprocess/Freiburg"
CROP_WIDTH = [300, 1700]
# Create output directories
os.makedirs(OUTPUT_PATH_FREIBURG, exist_ok=True)

def get_thermal_file(split, color_file):
    file_name = color_file.split(os.sep)[-1]
    if split == "test":
        file_name = file_name.replace("rgb", "ir")
        update_number = file_name.split('_')[-2][:-1]
        file_name = '_'.join(file_name.split('_')[:-2] + [update_number, file_name.split('_')[-1]])
    elif split == "train":
        file_name = file_name.replace("fl_rgb", "fl_ir_aligned")
    return os.sep.join(color_file.split(os.sep)[:-1]) + os.sep + file_name

def one_step(folders):
    nested = []
    for folder in folders:
        nested.extend([os.path.join(folder, fld) for fld in os.listdir(folder)])
    return nested

def convert_uint16_to_uint8(image):
    min_val, max_val = 21800, 25000 #obtained from the dataset github repository.
    image = np.clip(image, min_val, max_val)
    image = image.astype(np.float32)
    image -= min_val
    image /= (max_val - min_val + 1e-6) 

    # Scale to 0–255 and convert
    image *= 255.0
    return image.astype(np.uint8)

def process_freiburg(split, typ, color_folder, thermal_folder):
    split_path = os.path.join(RAW_DATA_PATH_FREIBURG, split)
    split_output_path = os.path.join(OUTPUT_PATH_FREIBURG, split, typ)

    # print(split_path, split_output_path)

    folders = [os.path.join(split_path, fld) for fld in os.listdir(split_path) if typ in fld]
    if split == "train":
        folders = one_step(folders)

    # print(folders)
    
    color_files, thermal_files = [], []
    for path in folders:
        color_path = os.path.join(path, color_folder)
        thermal_path = os.path.join(path, thermal_folder)
        color_files.extend([os.path.join(color_path, f) for f in sorted(os.listdir(color_path))])
        thermal_files.extend(set([os.path.join(thermal_path, f) for f in sorted(os.listdir(thermal_path))]))

    # print(len(color_files), len(thermal_files))
    # print(color_files[0], thermal_files[0])

    paired_samples = [] #list(zip(color_files, thermal_files))
    for color_file in color_files:
        thermal_file = get_thermal_file(split, color_file)  # Adjust this if naming differs
        thermal_file = thermal_file.replace(color_folder, thermal_folder)
        if thermal_file in thermal_files:
            paired_samples.append((color_file, thermal_file))
    
    # print(len(paired_samples))
    # print(paired_samples[0])

    shard_size = 1000
    num_samples = len(paired_samples)
    num_shards = (num_samples + shard_size - 1) // shard_size
    valid_count = 0

    with tqdm(total=num_samples, desc="Formatting Freiburg dataset") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(split_output_path, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for sample_idx in range(shard_idx * shard_size, min((shard_idx + 1) * shard_size, num_samples)):
                    rgb_path, ir_path = paired_samples[sample_idx]
                    sample_key = rgb_path.split(".")[0].split(os.sep)[-1]
                    try:
                        # Read IR image with OpenCV
                        ir_img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
                        ir_img = ir_img[:, CROP_WIDTH[0]:CROP_WIDTH[1]] # Crop black padding
                        # Detect black padding
                        if (0 in ir_img):
                            raise KeyError(f"Find invalid thermal pixels")
                        if ir_img is None:
                            raise ValueError(f"Failed to read IR image: {ir_path}")
                        # Convert uint16 to uint8.
                        ir_img = convert_uint16_to_uint8(ir_img)
                        # ir_img = cv2.convertScaleAbs(ir_img, alpha = 1./256., beta=-.49999)
                        # Re-encode as PNG
                        _, ir_image_encoded = cv2.imencode('.png', ir_img)
                        # Read RGB image with OpenCV
                        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                        rgb_img = rgb_img[:, CROP_WIDTH[0]:CROP_WIDTH[1]] # Crop black padding
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
                        valid_count += 1
                    except Exception as e:
                        print(f"Error processing sample {sample_key}: {e}")
                    pbar.update(1)
    print(f"Total samples formatted for Freiburg: {valid_count}")

    # Save the number of samples in a JSON file
    json_path = os.path.join(split_output_path, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": valid_count}, f)
    print(f"Metadata saved in {json_path}")
    
# Process both datasets
process_freiburg("test", "day", "ImagesRGB", "ImagesIR")
process_freiburg("test", "night", "ImagesRGB", "ImagesIR")
process_freiburg("train", "day", "fl_rgb", "fl_ir_aligned")
process_freiburg("train", "night", "fl_rgb", "fl_ir_aligned")

print("Dataset formatting completed.")

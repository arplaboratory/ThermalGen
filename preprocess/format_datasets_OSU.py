import os
import cv2
import glob
import json
import shutil
import numpy as np
from tqdm import tqdm
import webdataset as wds

OSU_PATH = "datasets_raw/OSU/OSU"
OUTPUT_PATH_OSU = "datasets_preprocess/OSU"

os.makedirs(OUTPUT_PATH_OSU, exist_ok=True)

def find_first_non_black_last_row_col(img):
    h, w = img.shape[:2]

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    coords = {}

    # Last row (left to right)
    last_row = gray[-1, :]
    for j, val in enumerate(last_row):
        if val > 0:
            coords['last_row_be'] = (h-1, j)
            break

    # Last row (right to left)
    for j in range(w-1, -1, -1):
        if last_row[j] > 0:
            coords['last_row_ed'] = (h-1, j)
            break

    # Last column (top to bottom)
    last_col = gray[:, coords["last_row_ed"][1]]
    for i, val in enumerate(last_col):
        if val > 0:
            coords['first_row_be'] = (i, w-1)
            break

    return coords

def crop(rgb_folders, thermal_folders):
    rgb_folders = ["1b", "2b", "3b", "4b", "5b", "6b"]
    thermal_folders = ["1a", "2a", "3a", "4a", "5a", "6a"]

    for (rgb_folder, thermal_folder) in zip(rgb_folders, thermal_folders):

        rgb_files = sorted(glob.glob(os.path.join(OSU_PATH, rgb_folder, "*.bmp")))
        thermal_files = sorted(glob.glob(os.path.join(OSU_PATH, thermal_folder, ".bmp")))

        rgb_cropped_path = os.path.join(OSU_PATH, f"{rgb_folder}_cropped")
        thermal_cropped_path = os.path.join(OSU_PATH, f"{thermal_folder}_cropped")

        if os.path.exists(rgb_cropped_path) and os.path.isdir(rgb_cropped_path):
            shutil.rmtree(rgb_cropped_path)

        if os.path.exists(thermal_cropped_path) and os.path.isdir(thermal_cropped_path):
            shutil.rmtree(thermal_cropped_path)

        os.makedirs(rgb_cropped_path, exist_ok=True)
        os.makedirs(thermal_cropped_path, exist_ok=True)

        is_first = True
        req_size = None
        for rgb_file, thermal_file in tqdm(zip(rgb_files, thermal_files), desc=f"Images cropped in {thermal_folder}"):
            img = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
            rgb_img = cv2.imread(thermal_file, cv2.IMREAD_UNCHANGED)
        
            # cropped = crop_colored_region_accurately(img)
            coordinates = find_first_non_black_last_row_col(img)
            cropped = img[coordinates['first_row_be'][0]:coordinates['last_row_be'][0]+1, coordinates['last_row_be'][1]:coordinates['last_row_ed'][1]+1]
            rgb_cropped = rgb_img[coordinates['first_row_be'][0]:coordinates['last_row_be'][0]+1, coordinates['last_row_be'][1]:coordinates['last_row_ed'][1]+1, :]

            if is_first:
                is_first = False
                req_size = cropped.shape
                cv2.imwrite(os.path.join(rgb_cropped_path, rgb_file), rgb_cropped)
                cv2.imwrite(os.path.join(thermal_cropped_path, thermal_file), cropped)
            elif cropped.shape[0] == req_size[0] and cropped.shape[1] == req_size[1] and cropped.shape[2] == req_size[2]:
                cv2.imwrite(os.path.join(rgb_cropped_path, rgb_file), rgb_cropped)
                cv2.imwrite(os.path.join(thermal_cropped_path, thermal_file), cropped)
            else:
                print(f"{rgb_file} and {thermal_file} not considered")

def process_osu(color_folders, thermal_folders):
    color_files, thermal_files = [], []
    for color_folder, thermal_folder in zip(color_folders, thermal_folders):
        color_files.extend(sorted(glob.glob(os.path.join(OSU_PATH, color_folder + "_cropped", "*.bmp"))))
        thermal_files.extend(sorted(glob.glob(os.path.join(OSU_PATH, thermal_folder + "_cropped", "*.bmp"))))

    paired_samples = list(zip(color_files, thermal_files))

    shard_size = 1000
    num_samples = len(paired_samples)
    num_shards = (num_samples + shard_size - 1) // shard_size

    with tqdm(total=num_samples, desc="Formatting OSU dataset") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(OUTPUT_PATH_OSU, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for sample_idx in range(shard_idx * shard_size, min((shard_idx + 1) * shard_size, num_samples)):
                    rgb_path, ir_path = paired_samples[sample_idx]
                    sample_key = os.sep.join(rgb_path.split(os.sep)[-2:]).split(".")[0].replace('/', '_')
                    try:
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
    print(f"Total samples formatted for OSU: {num_samples}")

    # Save the number of samples in a JSON file
    json_path = os.path.join(OUTPUT_PATH_OSU, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": num_samples}, f)
    print(f"Metadata saved in {json_path}")

if __name__=="__main__":

    rgb_folders = ["1b", "2b", "4b", "5b", "6b"]
    thermal_folders = ["1a", "2a", "4a", "5a", "6a"]

    # crop(rgb_folders, thermal_folders)

    process_osu(rgb_folders, thermal_folders)


    

import os
import webdataset as wds
from tqdm import tqdm
import cv2
import json

# Root path to raw KAIST data
RAW_DATA_PATH_KAIST = "datasets_raw/KAIST/KAIST-Multispectral-Pedestrian-Detection-Dataset/kaist_train"

# Output path for the processed, sharded dataset
OUTPUT_PATH_KAIST = "datasets_preprocess/KAIST"

def process_kaist():
    # Make sure the output directory exists
    os.makedirs(OUTPUT_PATH_KAIST, exist_ok=True)

    shard_size = 1000
    all_pairs = []

    # Recursively walk through the setXX/VXXX structure
    for set_name in sorted(os.listdir(RAW_DATA_PATH_KAIST)):
        set_path = os.path.join(RAW_DATA_PATH_KAIST, set_name)
        if not os.path.isdir(set_path):
            continue
        
        for v_name in sorted(os.listdir(set_path)):
            v_path = os.path.join(set_path, v_name)
            if not os.path.isdir(v_path):
                continue

            # Each "VXXX" directory should contain "visible" and "lwir" subfolders
            visible_path = os.path.join(v_path, "visible")
            lwir_path    = os.path.join(v_path, "lwir")

            # Only proceed if both subfolders exist
            if not os.path.isdir(visible_path) or not os.path.isdir(lwir_path):
                continue
            
            # Gather file names
            visible_files = sorted(os.listdir(visible_path))
            lwir_files_set = set(os.listdir(lwir_path))

            # Pair up files that match in both subfolders
            for vis_file in visible_files:
                # Here we assume the same filename is used in lwir (e.g. "I00000.jpg")
                # Adjust if the naming pattern differs.
                if vis_file in lwir_files_set:
                    visible_full_path = os.path.join(visible_path, vis_file)
                    lwir_full_path    = os.path.join(lwir_path, vis_file)
                    all_pairs.append((visible_full_path, lwir_full_path))

    # Create shards
    num_samples = len(all_pairs)
    num_shards = (num_samples + shard_size - 1) // shard_size

    with tqdm(total=num_samples, desc="Formatting KAIST dataset") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(OUTPUT_PATH_KAIST, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                start_idx = shard_idx * shard_size
                end_idx = min((shard_idx + 1) * shard_size, num_samples)
                
                for sample_idx in range(start_idx, end_idx):
                    vis_path, lwir_path = all_pairs[sample_idx]
                    # Create a unique key (you can also base this on subfolder names + filename)
                    sample_key = f"{shard_idx:05d}-{sample_idx:07d}"

                    try:
                        # Read LWIR
                        lwir_img = cv2.imread(lwir_path, cv2.IMREAD_UNCHANGED)
                        if lwir_img is None:
                            raise ValueError(f"Failed to read LWIR image: {lwir_path}")
                        _, lwir_encoded = cv2.imencode(".png", lwir_img)

                        # Read Visible
                        vis_img = cv2.imread(vis_path, cv2.IMREAD_UNCHANGED)
                        if vis_img is None:
                            raise ValueError(f"Failed to read Visible image: {vis_path}")
                        _, vis_encoded = cv2.imencode(".png", vis_img)

                        # Write to a shard in WebDataset format
                        sink.write({
                            "__key__":      sample_key,
                            "color.png":  vis_encoded.tobytes(),
                            "thermal.png":     lwir_encoded.tobytes(),
                        })
                    except Exception as e:
                        print(f"Error processing sample {sample_key}: {e}")
                    pbar.update(1)

    # Print summary
    print(f"Total samples formatted for KAIST: {num_samples}")

    # Save metadata (optional)
    json_path = os.path.join(OUTPUT_PATH_KAIST, "metadata.json")
    with open(json_path, "w") as f:
        json.dump({"num_samples": num_samples}, f)
    print(f"Metadata saved in {json_path}")

if __name__ == "__main__":
    process_kaist()
    print("KAIST dataset formatting completed.")

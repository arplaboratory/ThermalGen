import os
import webdataset as wds
from tqdm import tqdm
from PIL import Image
import io
import cv2
import json
import numpy as np

RAW_DATA_PATH_FLIR = "datasets_raw/FLIR/align/JPEGImages"
OUTPUT_PATH_FLIR   = "datasets_preprocess/FLIR"
os.makedirs(OUTPUT_PATH_FLIR, exist_ok=True)

def process_flir(split):
    """
    Args
    ----
    split : "train" or "validation"
    Expects RAW_DATA_PATH_FLIR/{split}.txt listing sample keys, one per line.
    """
    txt_file = os.path.join("datasets_raw/FLIR/align", f"align_{split}.txt")
    if not os.path.isfile(txt_file):
        raise FileNotFoundError(f"{txt_file} not found")
    with open(txt_file) as f:
        sample_keys = [ln.strip() for ln in f if ln.strip()]
    # ---------- write sharded WebDataset ----------
    out_dir   = os.path.join(OUTPUT_PATH_FLIR, split)
    os.makedirs(out_dir, exist_ok=True)
    shard_sz  = 1000
    num_shard = (len(sample_keys) + shard_sz - 1) // shard_sz

    with tqdm(total=len(sample_keys), desc=f"Formatting {split}") as pbar:
        for s in range(num_shard):
            shard_path = os.path.join(out_dir, f"dataset-{s:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                for idx in range(s*shard_sz, min((s+1)*shard_sz, len(sample_keys))):
                    ir_fn = sample_keys[idx] + ".jpeg"
                    rgb_fn = '_'.join(sample_keys[idx].split('_')[:2]) + "_RGB.jpg"
                    key = os.path.splitext(rgb_fn)[0]
                    try:
                        rgb = cv2.imread(os.path.join(RAW_DATA_PATH_FLIR,  rgb_fn), cv2.IMREAD_UNCHANGED)
                        ir  = cv2.imread(os.path.join(RAW_DATA_PATH_FLIR,   ir_fn),  cv2.IMREAD_UNCHANGED)
                        if rgb is None or ir is None:
                            raise ValueError("image read failed")

                        _, rgb_enc = cv2.imencode(".png", rgb)
                        _, ir_enc  = cv2.imencode(".png", ir)

                        sink.write({
                            "__key__": key,
                            "color.png":   rgb_enc.tobytes(),
                            "thermal.png": ir_enc.tobytes(),
                        })
                    except Exception as e:
                        print(f"⚠️  {key}: {e}")
                    pbar.update(1)

    # metadata
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"num_samples": len(sample_keys)}, f)
    print(f"Metadata written for {split}")

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
process_flir("train")
process_flir("validation")
print("Dataset formatting completed.")
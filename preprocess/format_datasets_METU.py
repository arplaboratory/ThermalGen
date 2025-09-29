import os
import cv2
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm
import json

def read_vistir(path, cam_K, dist):
    """
    Read and undistort either visible or thermal image with the provided intrinsics and distortion.
    Returns:
        image (np.ndarray): undistorted image
        new_K (np.ndarray): updated camera intrinsics after undistortion
    """
    image = cv2.imread(path)
    h, w = image.shape[:2]
    # update camera matrix
    new_K, _ = cv2.getOptimalNewCameraMatrix(cam_K, dist, (w, h), 0, (w, h))
    # undistort
    image = cv2.undistort(image, cam_K, dist, None, new_K)
    return image, new_K

def compute_homography(T_0to1, K0, K1):
    """
    Computes 3x3 homography from camera0 to camera1 using T_0to1 (4x4) and intrinsics K0, K1.
    A naive planar assumption is used (z=0 in camera0 coords).
    """
    R = T_0to1[:3, :3]
    # For a pure fronto-parallel assumption, ignoring translation’s effect on a plane at infinite distance:
    H = K1 @ R @ np.linalg.inv(K0)
    return H

def warp_image(src_img, H, out_shape):
    """
    Warp src_img to out_shape using the homography H.
    Returns the warped image and a warp-valid mask.
    """
    H_out, W_out = out_shape
    warped = cv2.warpPerspective(
        src_img, H, (W_out, H_out), flags=cv2.INTER_LINEAR, borderValue=0
    )
    # Create a valid mask by warping a white image
    valid_mask_src = np.ones_like(src_img, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        valid_mask_src, H, (W_out, H_out), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return warped, warped_mask

def crop_valid_region(img_ref, img_other, mask):
    """
    Crops both images to the bounding box of the intersection of mask>0.
    Returns cropped (img_ref, img_other, mask_cropped).
    """
    valid_intersection = mask > 0
    ys, xs, _ = np.where(valid_intersection)
    if len(ys) == 0:
        # No overlap
        return None, None, None
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    ref_cropped = img_ref[y_min:y_max+1, x_min:x_max+1]
    other_cropped = img_other[y_min:y_max+1, x_min:x_max+1]
    mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]
    return ref_cropped, other_cropped, mask_cropped

def process_metu_vistir_split(
    split: str,
    dataset_root: str,
    scene_info_dir: str,
    list_file: str,
    output_dir: str,
    shard_size: int = 1000
):
    """
    Reads the val/test scene list, loads each .npz, warps the VISIBLE image into the THERMAL frame,
    crops the valid overlapping region, then writes to WebDataset shards.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(list_file, "r") as f:
        scene_list = [line.strip() for line in f if line.strip()]

    all_pairs = []

    for scene_name in tqdm(scene_list):
        scene_npz = os.path.join(scene_info_dir, scene_name)
        if not os.path.isfile(scene_npz):
            print(f"[WARNING] .npz not found: {scene_npz}")
            continue
        
        data = np.load(scene_npz, allow_pickle=True)
        intrinsics = data["intrinsics"]
        distortion_coefs = data["distortion_coefs"]
        poses = data["poses"]
        img_names = data["image_paths"]

        N = len(poses)
        step = 2
        for i in range(0, N, step):
            idx0 = i
            idx1 = i + 1
            if idx1 >= N:
                break  # incomplete pair

            # intrinsics
            K_vis = np.array(intrinsics[idx0][0], dtype=float).reshape(3, 3)
            K_therm = np.array(intrinsics[idx1][1], dtype=float).reshape(3, 3)
            dist_vis = np.array(distortion_coefs[idx0][0], dtype=float)
            dist_therm = np.array(distortion_coefs[idx1][1], dtype=float)

            # poses
            T_vis = poses[idx0]    # (4,4)
            T_therm = poses[idx1]  # (4,4)

            # T_vis->therm = T_therm * inv(T_vis)
            T_vis_to_therm = np.matmul(T_therm, np.linalg.inv(T_vis))

            # read images
            img_vis_path = os.path.join(dataset_root, img_names[idx0][0])
            img_therm_path = os.path.join(dataset_root, img_names[idx1][1])
            vis_undist, K_vis_new = read_vistir(img_vis_path, K_vis, dist_vis)
            therm_undist, K_therm_new = read_vistir(img_therm_path, K_therm, dist_therm)

            # compute homography: visible -> thermal
            H_v2t = compute_homography(T_vis_to_therm, K_vis_new, K_therm_new)

            # warp visible image into thermal frame
            h_therm, w_therm = therm_undist.shape[:2]
            vis_warped, warp_mask = warp_image(vis_undist, H_v2t, (h_therm, w_therm))

            # crop overlap
            # ref = thermal, other = warped visible
            therm_crop, vis_crop, mask_crop = crop_valid_region(
                therm_undist, vis_warped, warp_mask
            )
            if therm_crop is None:
                # no overlap, skip
                continue

            # store data
            all_pairs.append({
                "thermal_img": therm_crop,
                "visible_img": vis_crop,
                "key": f"{scene_name.split('.')[0]}-{idx0:06d}-{idx1:06d}",
            })

    # Write to WebDataset shards
    total_samples = len(all_pairs)
    num_shards = (total_samples + shard_size - 1) // shard_size
    print(f"Preparing to write {total_samples} samples into {num_shards} shards.")

    sample_counter = 0
    with tqdm(total=total_samples, desc=f"Building {split} shards") as pbar:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(output_dir, f"dataset-{shard_idx:05d}.tar")
            with wds.TarWriter(shard_path) as sink:
                start_idx = shard_idx * shard_size
                end_idx = min((shard_idx + 1) * shard_size, total_samples)
                for i in range(start_idx, end_idx):
                    sample = all_pairs[i]
                    key = sample["key"]

                    # Encode as PNG
                    therm_encoded = cv2.imencode(".png", sample["thermal_img"])[1].tobytes()
                    vis_encoded = cv2.imencode(".png", sample["visible_img"])[1].tobytes()

                    sink.write({
                        "__key__": key,
                        "thermal.png": therm_encoded,
                        "color.png": vis_encoded
                    })
                    sample_counter += 1
                    pbar.update(1)

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"num_samples": sample_counter}, f)
    print(f"[{split}] Done. Total {sample_counter} samples. Metadata: {metadata_path}")


if __name__ == "__main__":
    dataset_root = "datasets_raw/METU_VisTIR/METU_VisTIR"
    val_scene_info_dir = os.path.join(dataset_root, "index", "scene_info_val")
    test_scene_info_dir = os.path.join(dataset_root, "index", "scene_info_test")
    val_list_file = os.path.join(dataset_root, "index", "val_test_list", "val_list.txt")
    test_list_file = os.path.join(dataset_root, "index", "val_test_list", "test_list.txt")

    output_val_dir = "datasets_preprocess/METU_VisTIR/val"
    output_test_dir = "datasets_preprocess/METU_VisTIR/test"

    # Process val (visible -> thermal)
    process_metu_vistir_split(
        split="val",
        dataset_root=dataset_root,
        scene_info_dir=val_scene_info_dir,
        list_file=val_list_file,
        output_dir=output_val_dir,
        shard_size=1000
    )

    # Process test (visible -> thermal)
    process_metu_vistir_split(
        split="test",
        dataset_root=dataset_root,
        scene_info_dir=test_scene_info_dir,
        list_file=test_list_file,
        output_dir=output_test_dir,
        shard_size=1000
    )

    print("All done!")

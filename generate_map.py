import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

from thermalgen import ThermalGen
from utils.load_cfg import load_config, load_datasets_config
from dataloaders.GenericDataloader import GenericDataModule
import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models

def create_center_weighted_mask(shape):
    """
    Create a center-weighted mask for an image patch.
    The weights decrease as you move away from the center.
    
    :param shape: Tuple (height, width, channels) of the patch.
    :return: Center-weighted mask.
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2

    # Calculate normalized distance from the center
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    weight = 1 - (distance_from_center / max_distance)

    # Normalize to range [0, 1]
    weight = np.clip(weight, 0, 1)

    # Expand to three channels
    return weight

def generate_map(model, datamodule, datafolder_name, dataset_index, database_names):
    model.eval()
    eval_dataloaders = datamodule.get_STGL_generate_dataloaders(datafolder_name, dataset_index, database_names)
    if not os.path.isdir("generate"):
        os.mkdir("generate")
    for eval_dataloader in eval_dataloaders:
        map_name = eval_dataloader.dataset.database_name
        print(f"Generating the map for {map_name}")
        save_map_path = os.path.join("generate", f"{map_name}_generated.png")
        # First round get the max size of the image
        image_sat = Image.open(eval_dataloader.dataset.database_folder_map_path)
        max_x, max_y = image_sat.size
        # Initialize canvas and weight canvases
        canvas = np.zeros((max_y, max_x), dtype=np.float32)
        weight_canvas = np.zeros((max_y, max_x), dtype=np.float32)

        for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
            # Compute features of all images (images contains queries, positives and negatives)
            batch_cuda = (batch[0].cuda(), batch[1], batch[2].cuda())
            result = model.prediction_step(batch=batch_cuda, batch_idx=batch_idx)
            for i in range(len(result[0])):
                generated_query = v2.ToPILImage()(v2.Resize(512)(result[0][i].cpu()))
                cood_y = int(result[1][0][i])
                cood_x = int(result[1][1][i])
                patch = np.array(generated_query)
                h, w = patch.shape
                offset = int(w//2), int(h//2)
                # Create center-weighted mask for the patch
                weight_mask = create_center_weighted_mask(patch.shape)

                # Add weighted patch to canvas
                top = max(0, cood_y-offset[1])
                left = max(0, cood_x-offset[0])
                bottom = min(max_y, cood_y+offset[1])
                right = min(max_x, cood_x+offset[0])
                # Handle out-of-bounds patches
                top_offset = top - (cood_y - offset[1])
                left_offset = left - (cood_x - offset[0])
                bottom_offset = 2 * offset[1] - ((cood_y + offset[1]) - bottom)
                right_offset = 2 * offset[0] - ((cood_x + offset[0]) - right)
                canvas[top:bottom, left:right] += patch[top_offset:bottom_offset, left_offset:right_offset].astype(np.float32) * weight_mask[top_offset:bottom_offset, left_offset:right_offset]
                weight_canvas[top:bottom, left:right] += weight_mask[top_offset:bottom_offset, left_offset:right_offset]

        # Normalize the canvas to handle overlaps
        weight_canvas = np.maximum(weight_canvas, 1e-8)  # Avoid division by zero
        blended_image = (canvas / weight_canvas).astype(np.uint8)

        # Save the large image
        large_image_pil = Image.fromarray(blended_image)
        large_image_pil.save(save_map_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    wandb_logger = WandbLogger(name=args.config.split('/')[-1].split('.')[0], entity="unistgl", project="ThermalGen")
    datamodule = GenericDataModule(
        datasets_folder=train_cfg.datasets.datasets_folder,
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
        mixed_precision=True if train_cfg.training.mixed_precision else False,
    )
    
    model = ThermalGen.load_from_checkpoint(
        train_cfg.training.load,
        #---- Encoder
        model_arch=train_cfg.model.model_arch,
        model_config=train_cfg.model.model_config,
        lr=train_cfg.training.optimizer["lr"],
        optimizer=train_cfg.training.optimizer["name"],
        weight_decay=train_cfg.training.optimizer["weight_decay"], # 0.001 for sgd and 0 for adam,
        momentum=train_cfg.training.optimizer["momentum"],
        lr_sched=train_cfg.training.scheduler["name"],
        lr_sched_args = train_cfg.training.scheduler["args"],

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=train_cfg.training.loss["name"],
        loss_config=train_cfg.training.loss["config"],
        validation_type=train_cfg.training.validation_type
    )

    generate_map(model=model, datamodule=datamodule, datafolder_name=train_cfg.generate.datafolder_name,
                 dataset_index=train_cfg.generate.dataset_index, database_names=train_cfg.generate.database_names)
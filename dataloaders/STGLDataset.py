from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision
from torchvision import transforms as T
from torchvision.transforms import v2
import numpy as np
import random
import tqdm
import os
import concurrent.futures
import time
import torchvision.transforms.functional as F
import yaml
from dataloaders.norm_params import IMAGENET_MEAN_STD, THERMAL_MEAN_STD, NORMAL_MEAN_STD
import torch.distributed as dist

default_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORMAL_MEAN_STD['mean'], std=NORMAL_MEAN_STD['std']),
])

default_thermal_transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=THERMAL_MEAN_STD['mean'], std=THERMAL_MEAN_STD['std']),
])

class STGLDataset(IterableDataset):
    def __init__(
            self,
            datasets_folder="./datasets_preprocess/STGL",
            split="train",
            image_size=512,
            transform=default_transform,
            dataset_name=None,
            database_name=0,
            queries_name=0,
            dataset_index=0,
            shuffle=False,
            num_samples_per_epoch=None,
            image_norm="normal",
    ):
        super(STGLDataset, self).__init__()
        self.transform = transform
        self.split = split
        self.image_size = image_size
        self.database_name = database_name
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index
        self.generate_map = False
        if queries_name is None:
            self.generate_map = True
            queries_name = database_name # For generation
        self.queries_name = queries_name
        self.shuffle = shuffle
        self.num_samples_per_epoch = num_samples_per_epoch
        if image_norm == "normal":
            image_mean_std = NORMAL_MEAN_STD
        elif image_norm == "imagenet":
            image_mean_std = IMAGENET_MEAN_STD
        else:
            raise NotImplementedError()
        thermal_mean_std = THERMAL_MEAN_STD
        self.mean_image = image_mean_std['mean']
        self.std_image = image_mean_std['std']
        self.mean_thermal = thermal_mean_std['mean']
        self.std_thermal = thermal_mean_std['std']
        with open(f"{datasets_folder}/folder_config.yml", "r") as f:
            folder_config = yaml.safe_load(f)
        database_data_name = database_name.split("_")[0]
        queries_data_name = queries_name.split("_")[0]
        database_map_index = int(database_name.split("_")[1])
        queries_map_index = int(queries_name.split("_")[1])
        self.database_folder_map_path = os.path.join(
            datasets_folder, "maps", folder_config[database_data_name]["name"], folder_config[database_data_name]["maps"][database_map_index]
        )
        self.queries_folder_map_path = os.path.join(
            datasets_folder, "maps", folder_config[queries_data_name]["name"], folder_config[queries_data_name]["maps"][queries_map_index]
        )
        
        self.queries_folder_coords, self.database_folder_coords = self.grid_sample(database_region=folder_config[database_data_name]["valid_regions"][database_map_index], 
                                                                    queries_region=folder_config[queries_data_name]["valid_regions"][queries_map_index],
                                                                    sample_width=512,
                                                                    stride=35 if not self.generate_map else 128,
                                                                    generate_map=self.generate_map)
        self.queries_folder_map_df = F.to_tensor(Image.open(self.queries_folder_map_path).convert("L"))
        self.database_folder_map_df = F.to_tensor(Image.open(self.database_folder_map_path).convert("RGB"))

    def calc_overlap(self, database_region, query_region):
        valid_region = []
        valid_region.append(max(database_region[0], query_region[0])) # top
        valid_region.append(max(database_region[1], query_region[1])) # left
        valid_region.append(min(database_region[2], query_region[2])) # bottom
        valid_region.append(min(database_region[3], query_region[3])) # right
        
        # Check if the region is valid
        if valid_region[2]<=valid_region[0] or valid_region[3]<=valid_region[1]:
            raise ValueError('The area of valid region is less or equal to zero.')
        
        # Check if the query region is inside the database region
        if query_region[0] < database_region[0] or query_region[1] < database_region[1] or query_region[2] > database_region[2] or query_region[3] > database_region[3]:
            raise ValueError('The query region is not inside the database region.')
            
        print("Get valid region: " + str(valid_region))
        return valid_region

    def grid_sample(self, database_region, queries_region, sample_width, stride, generate_map):
        valid_region = self.calc_overlap(database_region, queries_region)
        
        # Sample the valid region
        database_queries_region = [valid_region[0] + sample_width//2,
                                valid_region[1] + sample_width//2,
                                valid_region[2] - sample_width//2,
                                valid_region[3] - sample_width//2]  # top, left, bottom, right
        if generate_map:
            # For fully coverage
            database_queries_region[2] += sample_width//2
            database_queries_region[3] += sample_width//2
        cood_y_only = np.arange(
                    database_queries_region[0], database_queries_region[2], step=stride)
        cood_x_only = np.arange(
                    database_queries_region[1], database_queries_region[3], step=stride)
        cood_x, cood_y = np.meshgrid(cood_x_only, cood_y_only)
        cood_y = cood_y.flatten()
        cood_x = cood_x.flatten()
        queries_folder_coords = []
        database_folder_coords = []
        for i in range(len(cood_y)):
            queries_folder_coords.append((cood_y[i], cood_x[i]))
            database_folder_coords.append((cood_y[i], cood_x[i]))
        return queries_folder_coords, database_folder_coords    

    def _find_img_in_map(self, center_cood, database_queries_split):
        if database_queries_split == "database":
            img = self.database_folder_map_df
            width = self.image_size//2
        elif database_queries_split == "queries":
            img = self.queries_folder_map_df
            width = self.image_size//2 # avoid black padding
        area = (int(center_cood[1]) - width, int(center_cood[0]) - width,
                int(center_cood[1]) + width, int(center_cood[0]) + width)
        img = F.crop(img=img, top=area[1], left=area[0], height=area[3]-area[1], width=area[2]-area[0])
        return img

    def __iter__(self):
        if self.shuffle:
            # Zip them together into one list of pairs
            pairs = list(zip(self.queries_folder_coords, self.database_folder_coords))
            # Shuffle in-place
            random.shuffle(pairs)
        else:
            # No shuffle: directly zip them
            pairs = list(zip(self.queries_folder_coords, self.database_folder_coords))

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            pairs = pairs[rank::world_size]

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process or single-worker data loading
            for query_cood, database_cood in pairs:
                query, positive = self.transform(
                    v2.Grayscale(num_output_channels=1)(v2.ToImage()(self._find_img_in_map(query_cood, database_queries_split="queries"))),
                    v2.ToImage()(self._find_img_in_map(database_cood, database_queries_split="database"))
                )
                positive = v2.Normalize(mean=self.mean_image, std=self.std_image)(positive)
                query = v2.Normalize(mean=self.mean_thermal, std=self.std_thermal)(query)
                if not self.generate_map:
                    yield positive, query, self.dataset_index
                else:
                    yield positive, database_cood, self.dataset_index
        else:
            # Multi-worker data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Distribute the pairs among workers
            for i in range(worker_id, len(pairs), num_workers):
                query_cood, database_cood = pairs[i]
                query, positive = self.transform(
                    v2.Grayscale(num_output_channels=1)(v2.ToImage()(self._find_img_in_map(query_cood, database_queries_split="queries"))),
                    v2.ToImage()(self._find_img_in_map(database_cood, database_queries_split="database"))
                )
                positive = v2.Normalize(mean=self.mean_image, std=self.std_image)(positive)
                query = v2.Normalize(mean=self.mean_thermal, std=self.std_thermal)(query)
                if not self.generate_map:
                    yield positive, query, self.dataset_index
                else:
                    yield positive, database_cood, self.dataset_index

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {len(self.database_folder_coords)}; #queries: {len(self.queries_folder_coords)} >"

    def __len__(self):
        return len(self.database_folder_coords) if self.num_samples_per_epoch is None else self.num_samples_per_epoch
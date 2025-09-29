import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
import torch
import os
import numpy as np
import random
import webdataset as wds
import glob
import json
from torch.utils.data import IterableDataset

from dataloaders.STGLDataset import STGLDataset
from utils.load_cfg import load_datasets_config
import wandb
from dataloaders.norm_params import IMAGENET_MEAN_STD, THERMAL_MEAN_STD, NORMAL_MEAN_STD

class GenericDataModule(pl.LightningDataModule):
    def __init__(self,
                 datasets_folder="./datasets_preprocess/",
                 train_batch_size=8,
                 test_batch_size=16,
                 train_image_size=(512, 512),
                 num_workers=4,
                 thermal_mean_std=THERMAL_MEAN_STD,
                 dataset_names=None,
                 train_cfg_training=None,
                 mixed_precision=False,
                 image_norm="normal",
                 ):
        super().__init__()
        self.datasets_folder = datasets_folder
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_image_size = train_image_size
        self.num_workers = num_workers
        self.image_norm = image_norm
        if image_norm == "normal":
            image_mean_std = NORMAL_MEAN_STD
        elif image_norm == "imagenet":
            image_mean_std = IMAGENET_MEAN_STD
        else:
            raise NotImplementedError()
        self.mean_image = image_mean_std['mean']
        self.std_image = image_mean_std['std']
        self.mean_thermal = thermal_mean_std['mean']
        self.std_thermal = thermal_mean_std['std']
        self.train_dataset_names = dataset_names.train_datasets
        self.val_dataset_names = dataset_names.val_datasets
        self.test_dataset_names = dataset_names.test_datasets
        self.train_datasets_cfg = load_datasets_config(self.train_dataset_names)
        self.val_datasets_cfg = load_datasets_config(self.val_dataset_names)
        self.test_datasets_cfg = load_datasets_config(self.test_dataset_names)
        self.train_cfg_training = train_cfg_training

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(self.train_image_size, scale=(0.5, 1.0), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float16, scale=True) if mixed_precision else v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=self.mean_image, std=self.std_image) # Done inside
            ])
        
        self.val_transform = v2.Compose([
            v2.Resize(self.train_image_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float16, scale=True) if mixed_precision else v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=self.mean_image, std=self.std_image) # Done inside
            ])

        self.test_transform = v2.Compose([
            v2.Resize(self.train_image_size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float16, scale=True) if mixed_precision else v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=self.mean_image, std=self.std_image) # Done inside
            ])

        self.train_loader_config = {
            'batch_size': self.train_batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None, # Shuffle is done inside dataset
            }

        self.eval_loader_config = {
            'batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None, # Shuffle is done inside dataset
            }

        # eval loader generic need to configure num_workers separately
        self.eval_loader_config_generic = {
            'batch_size': self.test_batch_size,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None, # Shuffle is done inside dataset
            }
        
        # diverse input size needs batch size = 1
        self.eval_loader_config_generic_diverse = {
            'batch_size': 1,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': None, # Shuffle is done inside dataset
            }
        
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader (pitts_train, msls_train, ...etc)
            self.train_datasets = []
            for dataset_name in self.train_dataset_names:
                dataset_index = self.train_datasets_cfg[dataset_name].dataset_index
                dataset_configs = self.train_datasets_cfg[dataset_name].train
                if dataset_name.startswith("Boson") or dataset_name.startswith("DJI"):
                    dataset = self.get_STGL_dataset('train', dataset_configs, self.train_transform, dataset_index, dataset_name, num_samples_per_epoch=self.train_cfg_training.num_samples_per_epoch)
                else:
                    dataset = self.get_generic_dataset('train', dataset_configs, self.train_transform, dataset_index, dataset_name, num_samples_per_epoch=self.train_cfg_training.num_samples_per_epoch)
                self.train_datasets.append(dataset)

        if stage == 'fit' or stage == 'validate':
            self.val_datasets = []
            for dataset_name in self.val_dataset_names:
                dataset_index = self.val_datasets_cfg[dataset_name].dataset_index
                try:
                    dataset_configs = self.val_datasets_cfg[dataset_name].val
                except Exception as e:
                    print(dataset_name)
                    raise e
                if dataset_name.startswith("Boson") or dataset_name.startswith("DJI"):
                    dataset = self.get_STGL_dataset('val', dataset_configs, self.val_transform, dataset_index, dataset_name)
                else:
                    dataset = self.get_generic_dataset('val', dataset_configs, self.val_transform, dataset_index, dataset_name)
                self.val_datasets.append(dataset)

        if stage == 'fit' or stage == 'test':
            self.test_datasets = []
            for dataset_name in self.test_dataset_names:
                dataset_index = self.test_datasets_cfg[dataset_name].dataset_index
                try:
                    dataset_configs = self.test_datasets_cfg[dataset_name].test
                except Exception as e:
                    print(dataset_name)
                    raise e
                if dataset_name.startswith("Boson") or dataset_name.startswith("DJI"):
                    dataset = self.get_STGL_dataset('test', dataset_configs, self.test_transform, dataset_index, dataset_name)
                else:
                    dataset = self.get_generic_dataset('test', dataset_configs, self.test_transform, dataset_index, dataset_name)
                self.test_datasets.append(dataset)

        print({'train_datasets': self.train_datasets_cfg, 'val_datasets': self.val_datasets_cfg, 'test_datasets': self.test_datasets_cfg})

    def get_STGL_dataset(self, split, STGL_dataset_configs, transform, dataset_index, dataset_name, num_samples_per_epoch=None):
        dataset_list = []
        shuffle = True if split == 'train' else False
        for STGL_dataset_config in STGL_dataset_configs:
            dataset_list.append(STGLDataset(datasets_folder=os.path.join(self.datasets_folder, STGL_dataset_config['datafolder_name']), transform=transform,
                                                 split=split, database_name=STGL_dataset_config['database_name'], queries_name=STGL_dataset_config['queries_name'],
                                                 dataset_index=dataset_index, dataset_name=dataset_name,
                                                 shuffle=shuffle, num_samples_per_epoch=num_samples_per_epoch, image_norm=self.image_norm))
            print(dataset_list[-1])
        if len(dataset_list) > 1:
            if split == 'train':
                dataset = RandomChainDataset(dataset_list, num_samples_per_epoch=num_samples_per_epoch)
                dataset.dataset_name = dataset_name
                dataset.split = dataset_list[-1].split
            else:
                dataset = torch.utils.data.ChainDataset(dataset_list)
                dataset.dataset_name = dataset_name
                dataset.split = dataset_list[-1].split
        else:
            dataset = dataset_list[0]
        return dataset

    def get_STGL_generate_datasets(self, datafolder_name, dataset_index, database_names):
        dataset_list = []
        shuffle = False
        for database_name in database_names:
            dataset_list.append(STGLDataset(datasets_folder=os.path.join(self.datasets_folder, datafolder_name), transform=self.test_transform,
                                            split="test", database_name=database_name, queries_name=None,
                                            dataset_index=dataset_index, dataset_name=database_name,
                                            shuffle=shuffle, num_samples_per_epoch=None, image_norm=self.image_norm))
        return dataset_list
    
    def get_STGL_generate_dataloaders(self, datafolder_name, dataset_index, database_names):
        dataloader_list = []
        dataset_list = self.get_STGL_generate_datasets(datafolder_name, dataset_index, database_names)
        for dataset in dataset_list:
            dataloader_list.append(DataLoader(dataset=dataset, **self.eval_loader_config))
        return dataloader_list

    def get_generic_dataset(self, split, dataset_configs, transform, dataset_index, dataset_name, num_samples_per_epoch=None):
        dataset_list = []
        for dataset_config in dataset_configs:
            allshards = glob.glob(os.path.join(self.datasets_folder, dataset_config["datafolder_name"], "dataset-*.tar"))
            dataset = wds.WebDataset(allshards, nodesplitter=wds.split_by_node, resampled=True if split == 'train' else False, shardshuffle=True if split == 'train' else False)
            tuple_pattern = ["color.png","thermal.png", "__key__"]
            if split == 'train':
                dataset = dataset.shuffle(1000).decode("pil").to_tuple(*tuple_pattern).map(lambda sample: self.generic_preprocess(sample, transform, dataset_index)).with_epoch(num_samples_per_epoch).with_length(num_samples_per_epoch)
            else:
                with open(os.path.join(os.path.join(self.datasets_folder, dataset_config["datafolder_name"], "metadata.json"))) as f:
                    metadata = json.load(f)
                dataset = dataset.decode("pil").to_tuple(*tuple_pattern).map(lambda sample: self.generic_preprocess(sample, transform, dataset_index)).with_length(metadata["num_samples"])
                dataset.total_len = metadata["num_samples"]
            dataset.split = split
            dataset.dataset_name = dataset_name
            dataset.my_shard_num = len(allshards)
            dataset_list.append(dataset)
            try:
                print( f"< {dataset.__class__.__name__}, {dataset.dataset_name} - #database: {len(dataset)}; #queries: {len(dataset)} >")
            except TypeError as e:
                print( f"< {dataset.__class__.__name__}, {dataset.dataset_name} - #database: {dataset.total_len}; #queries: {dataset.total_len} >")
        if len(dataset_list) > 1:
            if split == 'train':
                dataset = RandomChainDataset(dataset_list, num_samples_per_epoch=num_samples_per_epoch)
                dataset.dataset_name = dataset_name
                dataset.split = dataset_list[-1].split
            else:
                dataset = torch.utils.data.ChainDataset(dataset_list)
                dataset.dataset_name = dataset_name
                dataset.split = dataset_list[-1].split
        else:
            dataset = dataset_list[0]
        if "diverse_size" in dataset_config and dataset_config["diverse_size"]:
            dataset.diverse_size = True
        return dataset
    
    def generic_preprocess(self, sample, transform, dataset_index):
        # sample[2] is key for debugging
        rgb, thermal = transform(v2.ToImage()(sample[0]), v2.Grayscale(num_output_channels=1)(v2.ToImage()(sample[1])))
        rgb = v2.Normalize(mean=self.mean_image, std=self.std_image)(rgb)
        thermal = v2.Normalize(mean=self.mean_thermal, std=self.std_thermal)(thermal)
        return rgb, thermal, dataset_index

    def train_dataloader(self):
        mixed_dataset = RandomChainDataset(self.train_datasets, num_samples_per_epoch=self.train_cfg_training.num_samples_per_epoch)
        train_dataloaders = DataLoader(dataset=mixed_dataset, **self.train_loader_config)
        return train_dataloaders

    def val_dataloader(self):
        val_dataloaders = []
        for index, val_dataset_name in enumerate(self.val_dataset_names):
            val_dataset = self.val_datasets[index]
            if val_dataset_name.startswith("Boson") or val_dataset_name.startswith("DJI"):
                val_dataloaders.append(DataLoader(
                    dataset=val_dataset, **self.eval_loader_config))
            else:
                val_dataloaders.append(DataLoader(
                    dataset=val_dataset, num_workers=max(min(val_dataset.my_shard_num, self.num_workers)//len(self.trainer.device_ids), 1), **self.eval_loader_config_generic))
        return val_dataloaders
    
    def test_dataloader(self):
        test_dataloaders = []
        for index, test_dataset_name in enumerate(self.test_dataset_names):
            test_dataset = self.test_datasets[index]
            if test_dataset_name.startswith("Boson") or test_dataset_name.startswith("DJI"):
                test_dataloaders.append(DataLoader(
                    dataset=test_dataset, **self.eval_loader_config))
            else:
                if hasattr(test_dataset, "diverse_size") and test_dataset.diverse_size:
                    test_dataloaders.append(DataLoader(
                        dataset=test_dataset, num_workers=max(min(test_dataset.my_shard_num, self.num_workers)//len(self.trainer.device_ids), 1), **self.eval_loader_config_generic_diverse))
                else:
                    test_dataloaders.append(DataLoader(
                        dataset=test_dataset, num_workers=max(min(test_dataset.my_shard_num, self.num_workers)//len(self.trainer.device_ids), 1), **self.eval_loader_config_generic))
        return test_dataloaders

class RandomChainDataset(torch.utils.data.ChainDataset):

    def __init__(self, datasets, num_samples_per_epoch=None, probs=None, longest=True):
        super().__init__(datasets)
        self.probs = probs
        self.longest = longest
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return self.random_samples(sources, self.probs, longest=self.longest)

    def random_samples(self, sources, probs=None, longest=False):
        """Yield samples randomly from multiple sources based on given probabilities.

        Args:
            sources (list): List of iterable sources to draw samples from.
            probs (list, optional): List of probabilities for each source. Defaults to None.
            longest (bool): If True, continue until all sources are exhausted. Defaults to False.

        Yields:
            Sample randomly selected from one of the sources.
        """
        if probs is None:
            probs = [1] * len(sources)
        else:
            probs = list(probs)
        while len(sources) > 0:
            cum = (np.array(probs) / np.sum(probs)).cumsum()
            r = random.random()
            i = np.searchsorted(cum, r)
            try:
                yield next(sources[i])
            except StopIteration:
                if longest:
                    del sources[i]
                    del probs[i]
                else:
                    break

    def __len__(self):
        if self.num_samples_per_epoch is None:
            total = 0
            for d in self.datasets:
                assert isinstance(
                    d, IterableDataset
                ), "ChainDataset only supports IterableDataset"
                total += len(d)  # type: ignore[arg-type]
            return total
        else:
            return self.num_samples_per_epoch
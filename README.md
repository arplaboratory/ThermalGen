# ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation

[![arXiv](https://img.shields.io/badge/arXiv-2301.01234-B31B1B.svg)](https://arxiv.org/abs/2301.01234)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Dataset-ThermalGen-blue.svg)](https://huggingface.co/datasets/xjh19972/ThermalGen-Dataset)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Model-ThermalGen-blue.svg)](https://huggingface.co/collections/xjh19972/thermalgen-models-68d863b498a89060c05fae2f)
[![Model](https://img.shields.io/badge/Model-ThermalGen-green.svg)](https://drive.google.com/file/d/13Og-MmrYu27AH51FI6EDtlAyGADvplxr/view?usp=drive_link)

This is the official repository for [ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation]().

Related works:  

```
TBD
```

## Setup

We provide an [`env.yml`](env.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
# Under ThermalGen root folder
conda env create -f env.yml
conda activate ThermalGen
```

## Simple Inference Demo

To quickly get started with ThermalGen, we provide a [`thermalgen_demo.py`](thermalgen_demo.py) script. This demo automatically downloads the model from Hugging Face and runs inference on a sample image:

```bash
# From the ThermalGen root directory
conda activate ThermalGen
python thermalgen_demo.py
```

## Datasets
For Satellite-Aerial datasets, please download the dataset file from this link: [Dataset]() and directly put the folder in ``datasets_preprocess``.

For other datasets, please follow the link put in ``datasets_preprocess`` folder to download the raw dataset accordingly and put them in ``datasets_raw`` folder. For preprocessing, please check ``preprocessing`` folder. For example, if you want to preprocess AVIID dataset, run

```
# Under ThermalGen root folder
conda activate ThermalGen
python preprocess/format_datasets_AVIID.py # Ensure that AVIID raw dataset is put in ./ThermalGen/datasets_raw/AVIID.
```

The script will automatically create a ``AVIID`` folder in ``dataset_preprocess`` and put the preproccessed file there.

## Pretrained weights

We provide [pretrained weights](https://drive.google.com/file/d/13Og-MmrYu27AH51FI6EDtlAyGADvplxr/view?usp=drive_link) for ThermalGen-B/2, ThermalGen-L/2, ThermalGen-XL/2, ThermalGen-L/2-concat. To run the model, put the checkpoint folder under ``./ThermalGen/checkpoints/`` folder.

## Evaluation
For the evaluation of datasets, please refer to running script ``scripts/test_thermal_generation.sbatch`` and config file ``configs/test``.

For example, if you want to run ThermalGen-L/2-concat model, you should have a ``sit_l2_concat`` checkpoint folder in ``./ThermalGen/checkpoints/``. Run
```
# Under ThermalGen root folder
conda activate ThermalGen
export CONFIG=configs/test/sit_cond/sit_l2_concat.yml
scripts/test_thermal_generation.sbatch
```

To run CFG, refer to the example code in [test_cfg.sh](test_cfg.sh)

## Training
For the evaluation of datasets, please refer to running script ``scripts/train_thermal_generation_long.sbatch`` and config file ``configs/train``.

For example, if you want to train ThermalGen-L/2-concat model, run
```
# Under ThermalGen root folder
conda activate ThermalGen
export CONFIG=configs/train/sit_cond/sit_l2_concat.yml
./scripts/train_thermal_generation_long.sbatch
```

Please find the example code in [train_generate.sh](train_generate.sh)

## Generate Stitched Thermal Map

In generate_map.py, we provide a script to stitch generated thermal images into a large map using a given satellite map.

For example, if you want to use the ThermalGen-L/2-concat model to generate satellite 1, 4, and 7 (indexes in datasets_preprocess/STGL/folder_config.yml), refer to configs/generate/sit_l2_concat_2.yml:

```
generate:
  datafolder_name: STGL
  dataset_index: 1
  database_names: 
    - satellite_1
    - satellite_4
    - satellite_7
```

Then run the following command:
```
# From the ThermalGen root folder
conda activate ThermalGen
export CONFIG=configs/generate/sit_l2_concat_2.yml
python generate_map.py --config CONFIG
```

The generated maps will be saved in the ./generate folder.

To add your own satellite map for generation, revise the satellite entry under datasets_preprocess/STGL/folder_config.yml:  
(1) Add the file name to satellite/maps.  
(2) Specify the generating region (top, left, bottom, right) in satellite/valid_regions.  
(3) Use the corresponding index in this config to update configs/generate/sit_l2_concat_2.yml.  
(4) Run the generate_map script.  

## Acknowledgement

We thank the folowing projects for their open-source code: [SiT](https://github.com/willisma/SiT), [LDM](https://github.com/CompVis/latent-diffusion).

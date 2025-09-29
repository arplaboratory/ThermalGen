import yaml
import argparse
import os

def load_config(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    for key, value in params.items():
        if isinstance(value, dict):
            params[key] = argparse.Namespace(**value)
    params = argparse.Namespace(**params)
    # Convert string to float
    if hasattr(params, 'training') and hasattr(params.training, 'optimizer'):
        params.training.optimizer['lr'] = float(params.training.optimizer['lr'])
    return params

def load_datasets_config(datasets):
    params = dict()
    for dataset in datasets:
        params[dataset] = load_config(os.path.join("configs", "datasets", f"{dataset}.yml"))
    return params
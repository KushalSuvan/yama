import json

import os
from pathlib import Path

def get_config():
    dir_path = os.path.abspath(os.path.dirname(__file__))
    config_path = str(Path(dir_path) / 'config.json')

    config = None

    with open(config_path, 'r') as f:
        config = json.load(f)

    assert config is not None, "ERROR: Failed to load config.json"

    return config

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path(os.getcwd()) / model_folder / model_filename)
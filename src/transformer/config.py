import json
from pathlib import Path
import os


def get_config():
    config=None

    file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(file_path, 'r') as f:
        config = json.load(f)

    return config

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)
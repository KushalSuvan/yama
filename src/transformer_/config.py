import json
import os
from pathlib import Path

def get_config():
    config = None

    file_path = str(Path(os.path.abspath(os.path.dirname(__file__))) / 'config.json')
    with open(file_path, 'r') as f:
        config = json.load(f)

    print(f"-----------\nCorpus: {config['corpus']}\nTranslation: {config['lang_src']} to {config['lang_tgt']}\n-----------")
    return config

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path(os.getcwd()) / model_folder / model_filename)

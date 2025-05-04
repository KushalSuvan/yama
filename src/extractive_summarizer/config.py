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
import os
from typing import Any, Optional

import hydra
from hydra import compose, initialize_config_dir

from artefact_nca.config.config_utils import setup_config


def load_config(config_path: Optional[str] = None, config_name = "default") -> Any:
    setup_config()
    if config_path is not None:
        config_path = os.path.abspath(config_path)
        config_dir, config_file = os.path.split(config_path)
        try:
            with initialize_config_dir(config_dir):
                _config = compose(config_name=config_file)
        except ValueError as e:
            _config = compose(config_name=config_file)
    else:
        config_dir = os.path.abspath(os.getcwd())
        print("SECOND")
        try:
            with initialize_config_dir(config_dir):
                _config = compose(config_name=config_name)
        except ValueError as e:
            _config = compose(config_name=config_name)
        print(_config)
    return _config

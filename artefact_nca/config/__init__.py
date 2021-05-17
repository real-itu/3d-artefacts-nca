import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from artefact_nca.config.config import *  # noqa

from artefact_nca.config.config_utils import (
    CONFIGS,
    add_config,
    add_configs,
    clear_configs,
    setup_config,
)
from artefact_nca.config.load import load_config

__all__ = [
    "CONFIGS",
    "clear_configs",
    "add_config",
    "add_configs",
    "setup_config",
    "load_config",
]

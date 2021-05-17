from typing import Any, Dict, List

from hydra.core.config_store import ConfigStore

CONFIGS: List[Dict[str, Any]] = []  # noqa

REQUIRED_CONFIG_KEYS = ["name", "node"]


def verify_config_dict(config) -> None:
    keys = list(config.keys())
    for req in REQUIRED_CONFIG_KEYS:
        if req not in keys:
            raise ValueError("key {} not in config!".format(req))


def clear_configs():
    CONFIGS = []  # noqa
    ConfigStore.repo = {}


def add_config(config_dict: Dict[str, Any]) -> None:
    verify_config_dict(config_dict)
    CONFIGS.append(config_dict)


def add_configs(list_of_config_dicts: List[Dict[str, Any]]) -> None:
    for d in list_of_config_dicts:
        add_config(d)


def setup_config():
    cs = ConfigStore.instance()
    for config in CONFIGS:
        cs.store(**config)

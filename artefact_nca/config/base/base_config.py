from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING  # Do not confuse with dataclass.MISSING
from pydantic.dataclasses import dataclass

from artefact_nca.config.config_utils import add_configs
from artefact_nca.config.torch import AdamConf, DataLoaderConf, ExponentialLRConf
from artefact_nca.config.tune.tune_config import TuneConfig

# This denotes the base config, inherit these config classes when you want to implement your own config
DEFAULTS = {
    "model_config": "default",
    "dataset_config": "default",
    "dataloader_config": "default",
    "optimizer_config": "default",
    "scheduler_config": "default",
    "logging_config": "default",
    "tune_config": "default",
}


def make_trainer_defaults(overrides=[]):
    override_keys = []
    for o in overrides:
        override_keys.append(list(o.keys())[0])
    for d in DEFAULTS:
        if d not in override_keys:
            overrides.append({d: DEFAULTS[d]})
    return overrides


@dataclass
class BaseModelConfig:
    _target_: str = MISSING


@dataclass
class BaseDatasetConfig:
    _target_: str = MISSING


@dataclass
class BaseLoggingConfig:
    checkpoint_path: str = "checkpoints"
    tensorboard_log_path: Optional[str] = None


@dataclass
class BaseTrainerConfig:
    defaults: List[Any] = field(default_factory=lambda: make_trainer_defaults())
    _target_: str = MISSING

    name: Optional[str] = None
    pretrained_path: Optional[str] = None
    visualize_output: bool = True
    use_cuda: bool = False
    device_id: int = 0
    early_stoppage: bool = False
    loss_threshold: float = -float("inf")
    batch_size: int = 32
    epochs: int = 100
    checkpoint_interval: int = 100
    num_samples: Optional[int] = None

    model_config: Any = MISSING
    dataset_config: Any = MISSING
    optimizer_config: Any = MISSING
    scheduler_config: Any = MISSING
    logging_config: Any = MISSING
    dataloader_config: Any = MISSING
    tune_config: Any = MISSING

    config: Any = field(default_factory=lambda: {})


config_defaults = [{"trainer": "default"}]


@dataclass
class DefaultConfig:

    defaults: List[Any] = field(default_factory=lambda: config_defaults)
    trainer: Any = MISSING


config_dicts: List[Dict[str, Any]] = [
    dict(group="trainer/model_config", name="default", node=BaseModelConfig),
    dict(group="trainer/dataset_config", name="default", node=BaseDatasetConfig),
    dict(group="trainer/dataloader_config", name="default", node=DataLoaderConf),
    dict(group="trainer/optimizer_config", name="default", node=AdamConf),
    dict(group="trainer/scheduler_config", name="default", node=ExponentialLRConf),
    dict(group="trainer/logging_config", name="default", node=BaseLoggingConfig),
    dict(group="trainer/tune_config", name="default", node=TuneConfig),
    dict(group="trainer", name="default", node=BaseTrainerConfig),
    dict(name="default", node=DefaultConfig),
]

add_configs(config_dicts)

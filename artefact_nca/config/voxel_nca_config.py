import dataclasses
from dataclasses import field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING  # Do not confuse with dataclass.MISSING
from omegaconf import II
from pydantic.dataclasses import dataclass

from artefact_nca.config.base import (
    BaseDatasetConfig,
    BaseModelConfig,
    BaseTrainerConfig,
    make_trainer_defaults,
)
from artefact_nca.config.config_utils import add_configs


@dataclass
class VoxelCAModelConfig(BaseModelConfig):
    _target_: str = "artefact_nca.model.voxel_ca_model.VoxelCAModel"
    alpha_living_threshold: float = 0.1
    cell_fire_rate: float = 0.5
    step_size: float = 1.0
    perception_requires_grad: bool = True
    # to be populated later for dataset
    living_channel_dim: Optional[int] = None
    num_hidden_channels: Any = II("trainer.num_hidden_channels")
    normal_std: float = 0.0002
    use_bce_loss: Any = II("trainer.use_bce_loss")
    use_normal_init: bool = True
    zero_bias: bool = True
    update_net_channel_dims: List[int] = dataclasses.field(
        default_factory=lambda: [32, 32]
    )


@dataclass
class VoxelCADatasetConfig(BaseDatasetConfig):
    _target_: str = "artefact_nca.dataset.voxel_dataset.VoxelDataset"
    entity_name: Optional[str] = None
    target_voxel: Optional[Any] = None
    target_color_dict: Optional[Dict[Any, Any]] = dataclasses.field(
        default_factory=lambda: None
    )
    target_unique_val_dict: Optional[Dict[Any, Any]] = dataclasses.field(
        default_factory=lambda: None
    )
    nbt_path: Optional[str] = None
    load_coord: List[int] = dataclasses.field(default_factory=lambda: [50, 10, 10])
    load_entity_config: Dict[Any, Any] = dataclasses.field(default_factory=lambda: {})
    spawn_at_bottom: bool = False
    use_random_seed_block: bool = False
    input_shape: Optional[List[int]] = None
    num_hidden_channels: Any = II("trainer.num_hidden_channels")
    half_precision: Any = II("trainer.half_precision")
    pool_size: int = 32
    padding_by_power: Optional[int] = None


trainer_defaults = [{"model_config": "voxel"}, {"dataset_config": "voxel"}]


@dataclass
class VoxelCATrainerConfig(BaseTrainerConfig):
    _target_: str = "artefact_nca.trainer.voxel_ca_trainer.VoxelCATrainer"
    defaults: List[Any] = field(
        default_factory=lambda: make_trainer_defaults(overrides=trainer_defaults)
    )
    name: Optional[str] = None
    num_hidden_channels: int = 12
    half_precision: bool = False
    min_steps: int = 48
    max_steps: int = 64
    damage: bool = False
    num_damaged: int = 2
    damage_radius_denominator: int = 5
    torch_seed: Optional[int] = None
    use_dataset: bool = True
    use_model: bool = True
    use_iou_loss: bool = True
    use_bce_loss: bool = False
    update_dataset: bool = True
    use_sample_pool: bool = True
    early_stoppage: bool = True
    loss_threshold: float = 0.005


config_defaults = [{"trainer": "voxel"}]


@dataclass
class VoxelCAConfig:

    defaults: List[Any] = field(default_factory=lambda: config_defaults)
    trainer: Any = MISSING


config_dicts: List[Dict[str, Any]] = [
    dict(group="trainer/model_config", name="voxel", node=VoxelCAModelConfig),
    dict(group="trainer/dataset_config", name="voxel", node=VoxelCADatasetConfig),
    dict(group="trainer", name="voxel", node=VoxelCATrainerConfig),
    dict(name="voxel", node=VoxelCAConfig),
]

add_configs(config_dicts)

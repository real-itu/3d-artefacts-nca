from dataclasses import field
from typing import Any, Dict, Optional

from omegaconf import II
from pydantic.dataclasses import dataclass


@dataclass
class TuneConfig:
    metric: str = "loss"
    mode: str = "min"
    num_samples: int = 1
    name: Optional[str] = II("trainer.name")
    checkpoint_freq: int = 100
    checkpoint_at_end: bool = True
    additional_config: Dict[str, Any] = field(default_factory=lambda: {})

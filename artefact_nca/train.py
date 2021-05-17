import sys
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from artefact_nca.config.config_utils import setup_config

from artefact_nca.trainer.voxel_ca_trainer import VoxelCATrainer


def train(cfg: DictConfig):
    if "trainer" not in cfg:
        cfg["trainer"] = {}
    else:
        cfg = OmegaConf.to_container(cfg)
        cfg["trainer"].pop("config")
    ct = VoxelCATrainer.from_config(config=cfg["trainer"])
    ct.train()


if __name__ == "__main__":
    setup_config()
    config_path = None
    config_name = None
    hydra_args = "hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled".split(" ")
    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_path = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
        sys.argv.extend(hydra_args)
        config_path = os.path.abspath(config_path)
        config_path, config_name = os.path.split(config_path)
        config_name = config_name.replace(".yaml", "")
    main_wrapper = hydra.main(config_path=config_path, config_name=config_name)
    main_wrapper(train)()

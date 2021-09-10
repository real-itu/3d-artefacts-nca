from __future__ import annotations

import abc
import os
import typing
from datetime import datetime
from typing import Any, Dict, Optional

import attr
import numpy as np
import torch
import tqdm
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from artefact_nca.config import load_config

# internal
from artefact_nca.utils.logging import TensorboardLogger
from artefact_nca.utils.utils import makedirs

def fullname(cls):
    module = cls.__module__
    return "{}.{}".format(module, cls.__name__)

@attr.s
class BaseTorchTrainer(metaclass=abc.ABCMeta):

    _config_group_ = "trainer"
    _config_name_ = "default"

    name: Optional[str] = attr.ib(default=None)
    pretrained_path: Optional[str] = attr.ib(default=None)
    visualize_output: bool = attr.ib(default=False)
    use_cuda: bool = attr.ib(default=False)
    device_id: int = attr.ib(default=0)
    early_stoppage: bool = attr.ib(default=False)
    loss_threshold: float = attr.ib(default=-float("inf"))
    batch_size: int = attr.ib(default=32)
    epochs: int = attr.ib(default=100)
    checkpoint_interval: int = attr.ib(default=100)
    num_samples: Optional[int] = attr.ib(default=None)
    # config
    model_config: Dict[str, Any] = attr.ib(default={})
    dataset_config: Dict[str, Any] = attr.ib(default={})
    optimizer_config: Dict[str, Any] = attr.ib(default={})
    scheduler_config: Dict[str, Any] = attr.ib(default={})
    logging_config: Dict[str, Any] = attr.ib(default={})
    dataloader_config: Dict[str, Any] = attr.ib(default={})
    tune_config: Dict[str, Any] = attr.ib(default={})
    # copy config for logging purposes
    config: Dict[str, Any] = attr.ib(default={})

    # variables that will be populateds
    current_iteration: int = attr.ib(init=False)
    checkpoint_path: str = attr.ib(init=False)
    tensorboard_log_path: str = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.setup_trainer()

    def setup_trainer(self):
        self.current_iteration = 0
        if self.name is None:
            self.name = self.__class__.__name__
        self.additional_tune_config = self.tune_config["additional_config"]
        self.tune_config = {
            **{
                k: self.tune_config[k]
                for k in self.tune_config
                if k != "additional_config"
            },
            **self.additional_tune_config,
        }
        self.config = OmegaConf.to_container(self.config)
        self.config["trainer"]["name"] = self.name
        self.device = torch.device(
            "cuda:{}".format(self.device_id) if self.use_cuda else "cpu"
        )
        self.setup()
        self.setup_logging_and_checkpoints()
        self._setup_dataset()
        self.setup_dataloader()
        self._setup_model()
        self._setup_optimizer()
        self.load(self.pretrained_path)
        self.setup_device()
        self.post_setup()

    @classmethod
    def from_config(
        cls, config_path: Optional[str] = None, config: Dict[str, Any] = {}
    ) -> BaseTorchTrainer:
        _config = load_config(config_path, config_name=cls._config_name_)
        _config["trainer"]["config"] = _config
        _config["trainer"]["_target_"] = fullname(cls)
        return instantiate(_config.trainer, _recursive_=False, **config)

    def tune(
        self, tune_config: Dict[str, Any] = {}, trainer_overrides: Dict[str, Any] = {}
    ):
        try:
            from artefact_nca.tune.tune_wrapper import TuneWrapper
        except ImportError:
            print("Cannot import ray or mlflow, make sure they are installed!")
            raise
        tune_config = {**self.tune_config, **tune_config}
        tune_wrapper = TuneWrapper(
            config=tune_config,
            trainer_config=self.config,
            trainer_overrides=trainer_overrides,
        )
        return tune_wrapper.tune()

    def setup(self):
        pass

    def setup_logging_and_checkpoints(self):
        self.checkpoint_path = makedirs(self.logging_config["checkpoint_path"])
        self.tensorboard_log_path = self.logging_config["tensorboard_log_path"]
        self.run_name = "{}_{}/".format(
            datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), self.name
        )
        self.base_checkpoint_path = self.make_checkpoint(makedirs(self.checkpoint_path))
        self.checkpoint_path = makedirs(
            os.path.join(self.base_checkpoint_path, "checkpoints")
        )
        if self.tensorboard_log_path is None:
            self.tensorboard_log_path = makedirs(
                os.path.join(self.base_checkpoint_path, "tensorboard_logs")
            )
        self.tensorboard_logger = TensorboardLogger(self.tensorboard_log_path)

    # setup helper functions
    def pre_dataset_setup(self):
        pass

    def post_dataset_setup(self):
        pass

    def setup_dataset(self):
        self.dataset = instantiate(self.dataset_config)

    def _setup_dataset(self):
        self.pre_dataset_setup()
        self.setup_dataset()
        self.post_dataset_setup()

    def setup_dataloader(self):
        self.dataloader = instantiate(
            self.dataloader_config, batch_size=self.batch_size, dataset=self.dataset
        )

    def pre_model_setup(self):
        pass

    def post_model_setup(self):
        pass

    def setup_model(self):
        self.model = instantiate(self.model_config)

    def _setup_model(self):
        self.pre_model_setup()
        self.setup_model()
        self.post_model_setup()

    def pre_optimizer_setup(self):
        pass

    def post_optimizer_setup(self):
        pass

    def setup_optimizer(self):
        self.optimizer = instantiate(
            self.optimizer_config, params=self.model.parameters()
        )
        self.scheduler = instantiate(self.scheduler_config, optimizer=self.optimizer)

    def _setup_optimizer(self):
        self.pre_optimizer_setup()
        self.setup_optimizer()
        self.post_optimizer_setup()

    def setup_device(self):
        if self.model is not None:
            self.model.device = self.device
            self.model.to(self.device)
        if self.dataset is not None:
            self.dataset.device = self.device
            self.dataset.to_device(self.device)

    def post_setup(self):
        pass

    def to_device(self, device):
        self.device = device
        self.setup_device()

    def make_checkpoint(self, path: str):
        path = os.path.join(path, self.run_name)
        return path

    def save(
        self,
        base_path: Optional[str] = None,
        step: Optional[int] = None,
        path_name: Optional[str] = None,
    ) -> str:
        checkpoint_path = self.checkpoint_path
        if base_path is not None:
            checkpoint_path = base_path
        if path_name is None:
            if step is None:
                step = self.current_iteration
            path = makedirs("{}/{}".format(checkpoint_path, step))
            torch_path = "{}/{}_iteration_{}.pt".format(path, self.name, step)
        else:
            path = makedirs("{}/{}".format(checkpoint_path, path_name))
            torch_path = "{}/{}.pt".format(path, self.name)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, torch_path)
        self.save_config(path)
        return path

    def save_config(self, path) -> str:
        yaml_str = OmegaConf.to_yaml(self.config)
        config_path = os.path.join(path, "{}.yaml".format(self.name))
        with open(config_path, "w") as f:
            f.write(yaml_str)
        return config_path

    def load(self, pretrained_path):
        if pretrained_path is not None:
            self.pretrained_path = os.path.abspath(pretrained_path)
            self.load_model(self.pretrained_path)

    def load_model(
        self, checkpoint_path: str, load_optimizer_and_scheduler: bool = True
    ):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and "scheduler" in checkpoint:
            if load_optimizer_and_scheduler:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def log_epoch(self, train_metrics, epoch):
        for metric in train_metrics:
            self.tensorboard_logger.log_scalar(
                train_metrics[metric], metric, step=epoch
            )

    def visualize(self, *args, **kwargs):
        """Visualize output
        """
        pass

    @abc.abstractmethod
    def train_iter(
        self, batch_size: int, iteration: Optional[int] = None
    ) -> Dict[Any, Any]:
        """
            Training iteration, specify learning process here
        """

    def pre_train(self):
        pass

    def post_train(self):
        pass

    def pre_train_iter(self):
        pass

    def post_train_iter(self, train_output: Dict[Any, Any]):
        pass

    def train(
        self, batch_size=None, epochs=None, checkpoint_interval=None, visualize=None
    ) -> typing.Dict[str, Any]:
        """Main training function, should call train_iter
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if epochs is not None:
            self.epochs = epochs
        if checkpoint_interval is not None:
            self.checkpoint_interval = checkpoint_interval
        self.pre_train()
        self.setup_logging_and_checkpoints()
        logger.info(
            "Follow tensorboard logs with: tensorboard --logdir {}".format(
                self.tensorboard_log_path
            )
        )
        self.setup_dataloader()
        bar = tqdm.tqdm(np.arange(self.epochs))
        for i in bar:
            self.pre_train_iter()
            output = self.train_iter(self.batch_size, i)
            self.post_train_iter(output)
            metrics = output.get("metrics", {})
            loss = output["loss"]
            self.log_epoch(metrics, i)

            description = "--".join(["{}:{}".format(k, metrics[k]) for k in metrics])
            bar.set_description(description)
            if i % self.checkpoint_interval == 0:
                if self.visualize_output:
                    self.visualize(output)
                self.save(step=i)
            if self.early_stoppage:
                if loss <= self.loss_threshold:
                    break
        self.post_train()
        return metrics

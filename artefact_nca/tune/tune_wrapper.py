import os
from typing import Any, List

import attr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.trial import Trial


class CustomMLflowLoggerCallback(MLflowLoggerCallback):
    def __init__(self, *args, **kwargs):
        super(CustomMLflowLoggerCallback, self).__init__(*args, **kwargs)

    def on_trial_save(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        """Called after receiving a checkpoint from a trial.

        Arguments:
            iteration (int): Number of iterations of the tuning loop.
            trials (List[Trial]): List of trials.
            trial (Trial): Trial that just saved a checkpoint.
            **info: Kwargs dict for forward compatibility.
        """
        run_id = self._trial_runs[trial]

        # Log the artifact if set_artifact is set to True.
        self.client.log_artifacts(run_id, local_dir=trial.logdir)


class TuneTrainer(tune.Trainable):
    def setup(self, config):
        overrides = config.get("overrides", {})
        trainer_config = OmegaConf.structured(config.get("config"))
        trainer_config.trainer.config = trainer_config
        self.trainer = instantiate(
            trainer_config.trainer, _recursive_=False, **overrides
        )

    def step(self):
        out = self.trainer.train_iter(
            self.trainer.batch_size, self.trainer.current_iteration
        )
        self.trainer.current_iteration += 1
        return out["metrics"]

    def save_checkpoint(self, tmp_checkpoint_dir):
        return self.trainer.save(tmp_checkpoint_dir)

    def load_checkpoint(self, tmp_checkpoint_dir):
        self.trainer.load(
            os.path.join(tmp_checkpoint_dir, "{}.pt".format(self.trainer.name))
        )


def create_stopper(config):
    epochs = config.get("epochs", 1000)
    loss_threshold: float = -float("inf")
    if config.get("early_stoppage"):
        loss_threshold = config.get("loss_threshold", 0.005)

    def stopper(trial_id, result):
        if result["training_iteration"] >= epochs:
            return True
        return result["loss"] < loss_threshold

    return stopper


@attr.s
class TuneWrapper:

    config: Any = attr.ib()
    trainer_config: Any = attr.ib()
    trainer_overrides: Any = attr.ib()

    def tune(self):
        mlflow_callback = CustomMLflowLoggerCallback(
            experiment_name=self.trainer_config["trainer"]["name"], save_artifact=False
        )
        callbacks = self.config.get("callbacks", [])
        callbacks.append(mlflow_callback)
        self.config["callbacks"] = callbacks
        if "stop" not in self.config:
            self.config["stop"] = create_stopper(self.trainer_config["trainer"])
        return tune.run(
            TuneTrainer,
            config={"config": self.trainer_config, "overrides": self.trainer_overrides},
            **self.config
        )

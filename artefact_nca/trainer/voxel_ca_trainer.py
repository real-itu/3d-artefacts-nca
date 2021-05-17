from enum import Enum
from typing import Any, Dict, List, Optional

import attr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from hydra.utils import instantiate
from IPython.display import clear_output
from tqdm import tqdm

# internal
from artefact_nca.base.base_torch_trainer import BaseTorchTrainer
from artefact_nca.dataset.voxel_dataset import VoxelDataset
from artefact_nca.model.voxel_ca_model import VoxelCAModel
from artefact_nca.utils.minecraft import *  # noqa
from artefact_nca.utils.minecraft.voxel_utils import replace_colors


# zero out a cube
def damage_cube(state, x, y, z, half_width):

    damaged = state.clone()

    x_dim = state.shape[1]
    y_dim = state.shape[2]
    z_dim = state.shape[3]

    from_x = np.clip(x - half_width, a_min=0, a_max=x_dim)
    from_y = np.clip(y - half_width, a_min=0, a_max=y_dim)
    from_z = np.clip(z - half_width, a_min=0, a_max=z_dim)

    to_x = np.clip(x + half_width, a_min=0, a_max=x_dim)
    to_y = np.clip(y + half_width, a_min=0, a_max=y_dim)
    to_z = np.clip(z + half_width, a_min=0, a_max=z_dim)

    damaged[:, from_x:to_x, from_y:to_y, from_z:to_z, :] = 0
    return damaged


# zero out a sphere
def damage_sphere(state, x, y, z, radius):

    damaged = state.clone()

    x_ = list(range(state.shape[1]))
    y_ = list(range(state.shape[2]))
    z_ = list(range(state.shape[3]))

    x_, y_, z_ = np.meshgrid(x_, y_, z_, indexing="ij")

    dist_squared = (x_ - x) * (x_ - x) + (y_ - y) * (y_ - y) + (z_ - z) * (z_ - z)
    not_in_sphere = dist_squared > (radius * radius)
    damaged = state * not_in_sphere
    return damaged


@attr.s
class VoxelCATrainer(BaseTorchTrainer):

    _config_name_: str = "voxel"

    damage_radius_denominator: int = attr.ib(default=5)
    use_iou_loss: bool = attr.ib(default=True)
    use_bce_loss: bool = attr.ib(default=False)
    use_sample_pool: bool = attr.ib(default=True)
    num_hidden_channels: Optional[int] = attr.ib(default=12)
    num_categories: Optional[int] = attr.ib(default=None)
    use_dataset: bool = attr.ib(default=True)
    use_model: bool = attr.ib(default=True)
    half_precision: bool = attr.ib(default=False)
    min_steps: int = attr.ib(default=48)
    max_steps: int = attr.ib(default=64)
    damage: bool = attr.ib(default=False)
    num_damaged: int = attr.ib(default=2)
    torch_seed: Optional[int] = attr.ib(default=None)
    update_dataset: bool = attr.ib(default=True)
    seed: Optional[Any] = attr.ib(default=None)

    def post_dataset_setup(self):
        self.seed = self.dataset.get_seed(1)
        self.num_categories = self.dataset.num_categories
        self.num_channels = self.dataset.num_channels
        self.model_config["living_channel_dim"] = self.num_categories

    def get_seed(self, batch_size=1):
        return self.dataset.get_seed(batch_size)

    def sample_batch(self, batch_size: int):
        return self.dataset.sample(batch_size)

    def log_epoch(self, train_metrics, epoch):
        for metric in train_metrics:
            self.tensorboard_logger.log_scalar(
                train_metrics[metric], metric, step=epoch
            )

    def rank_loss_function(self, x, targets):
        x = rearrange(x, "b d h w c -> b c d h w").to(self.device)
        out = torch.mean(
            F.cross_entropy(
                x[:, : self.num_categories, :, :, :].float(), targets, reduction="none"
            ),
            dim=[-2, -3, -1],
        )
        return out

    def apply_damage(self, batch, coords=None, num_damaged=None):

        x_ = list(range(batch.shape[1]))
        y_ = list(range(batch.shape[2]))
        z_ = list(range(batch.shape[3]))
        r = np.max(batch.shape) // self.damage_radius_denominator
        if num_damaged is None:
            num_damaged = self.num_damaged
        for i in range(1, num_damaged + 1):
            if coords is None:
                x = np.random.randint(0, batch.shape[1])
                y = np.random.randint(0, batch.shape[2])
                z = np.random.randint(0, batch.shape[3])
            center = [x, y, z]
            coords = np.ogrid[: batch.shape[1], : batch.shape[2], : batch.shape[3]]
            distance = np.sqrt(
                (coords[0] - center[0]) ** 2
                + (coords[1] - center[1]) ** 2
                + (coords[2] - center[2]) ** 2
            )
            not_in_sphere = 1 * (distance > r)
            ones = (
                np.ones((batch.shape[1], batch.shape[2], batch.shape[3]))
                - not_in_sphere
            )
            batch[-i] *= torch.from_numpy(not_in_sphere[:, :, :, None]).to(self.device)
            batch[-i][:, :, :, 0] += torch.from_numpy(ones).to(self.device)
        return batch

    def visualize(self, out):
        prev_batch = out["prev_batch"]
        post_batch = out["post_batch"]
        prev_batch = rearrange(prev_batch, "b d h w c -> b w d h c")
        post_batch = rearrange(post_batch, "b d h w c -> b w d h c")
        prev_batch = replace_colors(
            np.argmax(prev_batch[:, :, :, :, : self.num_categories], -1),
            self.dataset.target_color_dict,
        )
        post_batch = replace_colors(
            np.argmax(post_batch[:, :, :, :, : self.num_categories], -1),
            self.dataset.target_color_dict,
        )
        clear_output()
        vis0 = prev_batch[:5]
        vis1 = post_batch[:5]
        num_cols = len(vis0)
        vis0[vis0 == "_empty"] = None
        vis1[vis1 == "_empty"] = None
        print("Before --- After")
        fig = plt.figure(figsize=(15, 10))
        for i in range(1, num_cols + 1):
            ax0 = fig.add_subplot(1, num_cols, i, projection="3d")
            ax0.voxels(vis0[i - 1], facecolors=vis0[i - 1], edgecolor="k")
            ax0.set_title("Index {}".format(i))
        for i in range(1, num_cols + 1):
            ax1 = fig.add_subplot(2, num_cols, i + num_cols, projection="3d")
            ax1.voxels(vis1[i - 1], facecolors=vis1[i - 1], edgecolor="k")
            ax1.set_title("Index {}".format(i))
        plt.subplots_adjust(bottom=0.005)
        plt.show()

    def rollout(self, initial=None, steps=100):
        if initial is None:
            initial, _, _ = self.sample_batch(1)
        if not isinstance(initial, torch.Tensor):
            initial = torch.from_numpy(initial).to(self.device)
        bar = tqdm(np.arange(steps))
        out = [initial]
        life_masks = [None]
        for i in bar:
            x, life_mask = self.model(out[-1], 1, return_life_mask=True)
            life_masks.append(life_mask)
            out.append(x)
        return out[-1], out, life_masks

    def update_dataset_function(self, out, indices):
        with torch.no_grad():
            if self.half_precision:
                self.dataset.data[indices, :] = out.detach().type(torch.float16)
            else:
                self.dataset.data[indices, :] = out.detach()

    def iou(self, out, targets):
        targets = torch.clamp(targets, min=0, max=1)
        out = torch.clamp(
            torch.argmax(out[:, : self.num_categories, :, :, :], 1), min=0, max=1
        )
        intersect = torch.sum(out & targets).float()
        union = torch.sum(out | targets).float()
        o = (union - intersect) / (union + 1e-8)
        return o

    def get_loss(self, x, targets):
        iou_loss = 0
        if self.use_iou_loss:
            iou_loss = self.iou(x, targets)
        if self.use_bce_loss:
            class_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :], targets, ignore_index=0
            )
            alive = torch.clip(x[:, self.num_categories, :, :, :], 0.0, 1.0)
            alive_target_cells = torch.clip(targets, 0, 1).float()
            alive_loss = torch.nn.MSELoss()(alive, alive_target_cells)
        else:
            class_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :], targets, ignore_index=0
            )
            weight = torch.zeros(self.num_categories)
            weight[0] = 1.0
            alive_loss = F.cross_entropy(
                x[:, : self.num_categories, :, :, :],
                targets,
                weight=weight.to(self.device),
            )
        loss = (0.5 * class_loss + 0.5 * alive_loss + iou_loss) / 3.0
        return loss, iou_loss

    def get_loss_for_single_instance(self, x, rearrange_input=False):
        if rearrange_input:
            x = rearrange(x, "b d h w c -> b c d h w")
        batch, targets, indices = self.sample_batch(1)
        return self.get_loss(x, targets)

    def train_func(self, x, targets, steps=1):
        self.optimizer.zero_grad()
        x = self.model(x, steps=steps, rearrange_output=False)
        loss, iou_loss = self.get_loss(x, targets)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        x = rearrange(x, "b c d h w -> b d h w c")
        out = {
            "out": x,
            "metrics": {"loss": loss.item(), "iou_loss": iou_loss.item()},
            "loss": loss,
        }
        return out

    def train_iter(self, batch_size=32, iteration=0):
        batch, targets, indices = self.sample_batch(batch_size)
        if self.use_sample_pool:
            with torch.no_grad():
                loss_rank = (
                    self.rank_loss_function(batch, targets)
                    .detach()
                    .cpu()
                    .numpy()
                    .argsort()[::-1]
                )
                batch = batch[loss_rank.copy()]
                batch[:1] = torch.from_numpy(self.get_seed()).to(self.device)

                if self.damage:
                    self.apply_damage(batch)

        steps = np.random.randint(self.min_steps, self.max_steps)
        if self.half_precision:
            with torch.cuda.amp.autocast():
                out_dict = self.train_func(batch, targets, steps)
        else:
            out_dict = self.train_func(batch, targets, steps)
        out, loss, metrics = out_dict["out"], out_dict["loss"], out_dict["metrics"]

        if self.update_dataset and self.use_sample_pool:
            self.update_dataset_function(out, indices)
        out_dict["prev_batch"] = batch.detach().cpu().numpy()
        out_dict["post_batch"] = out.detach().cpu().numpy()
        return out_dict

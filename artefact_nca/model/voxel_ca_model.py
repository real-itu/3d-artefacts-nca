import typing
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# internal
from artefact_nca.base.base_torch_model import BaseTorchModel


def make_sequental(num_channels, channel_dims):
    conv3d = torch.nn.Conv3d(num_channels * 3, channel_dims[0], kernel_size=1)
    relu = torch.nn.ReLU()
    layer_list = [conv3d, relu]
    for i in range(1, len(channel_dims)):
        layer_list.append(
            torch.nn.Conv3d(channel_dims[i - 1], channel_dims[i], kernel_size=1)
        )
        layer_list.append(torch.nn.ReLU())
    layer_list.append(
        torch.nn.Conv3d(channel_dims[-1], num_channels, kernel_size=1, bias=False)
    )
    return torch.nn.Sequential(*layer_list)


class VoxelPerceptionNet(torch.nn.Module):
    def __init__(
        self, num_channels, normal_std=0.02, use_normal_init=True, zero_bias=True
    ):
        super(VoxelPerceptionNet, self).__init__()
        self.num_channels = num_channels
        self.normal_std = normal_std
        self.conv1 = torch.nn.Conv3d(
            self.num_channels,
            self.num_channels * 3,
            3,
            stride=1,
            padding=1,
            groups=self.num_channels,
            bias=False,
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)

    def forward(self, x):
        return self.conv1(x)


class SmallerVoxelUpdateNet(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 16,
        channel_dims=[64, 64],
        normal_std=0.02,
        use_normal_init=True,
        zero_bias=True,
    ):
        super(SmallerVoxelUpdateNet, self).__init__()
        self.out = make_sequental(num_channels, channel_dims)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)

    def forward(self, x):
        return self.out(x)


class VoxelCAModel(BaseTorchModel):
    def __init__(
        self,
        alpha_living_threshold: float = 0.1,
        cell_fire_rate: float = 0.5,
        step_size: float = 1.0,
        perception_requires_grad: bool = True,
        living_channel_dim: Optional[int] = None,
        num_hidden_channels: Optional[int] = None,
        normal_std: float = 0.0002,
        use_bce_loss: bool = False,
        use_normal_init: bool = True,
        zero_bias: bool = True,
        update_net_channel_dims: typing.List[int] = [32, 32],
    ):
        super(VoxelCAModel, self).__init__()
        self.num_hidden_channels = num_hidden_channels
        self.update_net_channel_dims = update_net_channel_dims
        self.living_channel_dim = living_channel_dim
        self.num_categories = self.living_channel_dim
        self.alpha_living_threshold = alpha_living_threshold
        self.cell_fire_rate = cell_fire_rate
        self.step_size = step_size
        self.perception_requires_grad = perception_requires_grad
        self.normal_std = normal_std
        self.use_bce_loss = use_bce_loss
        self.use_normal_init = use_normal_init
        self.zero_bias = zero_bias
        self.num_channels = self.num_hidden_channels + self.num_categories + 1
        self.perception_net = VoxelPerceptionNet(
            self.num_channels,
            normal_std=self.normal_std,
            use_normal_init=self.use_normal_init,
            zero_bias=self.zero_bias,
        )
        if not self.perception_requires_grad:
            for p in self.perception_net.parameters():
                p.requires_grad = False
        self.update_network = SmallerVoxelUpdateNet(
            self.num_channels,
            channel_dims=self.update_net_channel_dims,
            normal_std=self.normal_std,
            use_normal_init=self.use_normal_init,
            zero_bias=self.zero_bias,
        )
        self.tanh = torch.nn.Tanh()

    def alive(self, x):
        return F.max_pool3d(
            x[:, self.living_channel_dim : self.living_channel_dim + 1, :, :, :],
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def perceive(self, x):
        return self.perception_net(x)

    def update(self, x):
        pre_life_mask = self.alive(x) > self.alpha_living_threshold

        out = self.perceive(x)
        out = self.update_network(out)

        rand_mask = torch.rand_like(x[:, :1, :, :, :]) < self.cell_fire_rate
        out = out * rand_mask.float().to(self.device)
        x = x + out

        post_life_mask = self.alive(x) > self.alpha_living_threshold
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        if not self.use_bce_loss:
            x[:, :1, :, :, :][life_mask == 0.0] += torch.tensor(1.0).to(self.device)
        return x, life_mask

    def forward(self, x, steps=1, rearrange_output=True, return_life_mask=False):
        x = rearrange(x, "b d h w c -> b c d h w")
        for step in range(steps):
            x, life_mask = self.update(x)
        if rearrange_output:
            x = rearrange(x, "b c d h w -> b d h w c")
        if return_life_mask:
            return x, life_mask
        return x

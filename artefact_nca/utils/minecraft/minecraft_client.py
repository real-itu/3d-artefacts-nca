import copy
import time

import grpc
import numpy as np
import torch
from einops import rearrange
from test_evocraft_py.minecraft_pb2 import *
from test_evocraft_py.minecraft_pb2_grpc import *
from tqdm import tqdm

from artefact_nca.utils.minecraft.block_loader import *


def spawn_entities(
    client_list,
    steps,
    unique_val_dict,
    initial_states=[None],
    block_priority=[REDSTONE_BLOCK],
    increments=5,
    wait=2,
):
    unique_val_dict = {int(k): unique_val_dict[k] for k in unique_val_dict}
    time.sleep(wait)
    out_arr = [None] * len(initial_states)
    block_arr = [None] * len(initial_states)
    for i in range(len(initial_states)):
        block_copy, out = client_list[i].spawn_steps(
            increments,
            unique_val_dict,
            out=initial_states[i],
            place_block_priority_first=False,
            clear_blocks=True,
        )
        out_arr[i] = out.clone()
    time.sleep(wait)
    for i in range(increments, steps, increments):
        for j in range(len(client_list)):
            block_list, o = client_list[j].spawn_steps(
                increments,
                unique_val_dict,
                initial=out_arr[j],
                place_block_priority_first=False,
                clear_blocks=False,
            )
            out_arr[j] = o.clone()

        time.sleep(0.8)
    for j in range(len(client_list)):
        block_list, o = client_list[j].spawn_steps(
            increments,
            unique_val_dict,
            out=out_arr[j],
            place_block_priority_first=True,
            clear_blocks=True,
        )
        out_arr[j] = o.clone()


entity_function_dict = {"flying_machine": create_flying_machine}


class MinecraftClient:
    @classmethod
    def load_entity(
        cls,
        entity_name=None,
        nbt_path=None,
        load_coord=(50, 10, 10),
        load_entity_config={},
    ):
        if entity_name is None and nbt_path is None:
            raise ValueError("Must provide entity_name and nbt_path")
        if nbt_path is not None:
            return read_nbt_target(
                nbt_path=nbt_path, load_coord=(50, 10, 10), **load_entity_config
            )
        else:
            if entity_name not in entity_function_dict:
                raise ValueError("Entity not in entity_function_dict")
            return entity_function_dict[entity_name](
                load_coord=load_coord, **load_entity_config
            )

    def __init__(self, trainer, coords, area_size=30, client=None):
        self.trainer = trainer
        self.model = trainer.model
        self.client = client
        if client is None:
            self.client = MinecraftServiceStub(grpc.insecure_channel("localhost:5001"))
        self.coords = coords
        self.min_bounds = (
            self.coords[0] - area_size // 2,
            self.coords[1],
            self.coords[2] - area_size // 2,
        )
        self.max_bounds = (
            self.coords[0] + area_size // 2,
            self.coords[1] + area_size,
            self.coords[2] + area_size // 2,
        )

    def create_block(
        self,
        index,
        block_type,
        block_orientation_dict={
            NORTH: [REDSTONE_BLOCK, PISTON, SLIME, QUARTZ_BLOCK],
            SOUTH: [STICKY_PISTON],
        },
    ):
        min_bounds = self.min_bounds
        x, y, z = index
        orientation = NORTH
        if block_type == 181:
            block_type = REDSTONE_BLOCK
        for k in block_orientation_dict:
            if block_type in block_orientation_dict[k]:
                orientation = k
        return Block(
            position=Point(
                x=x + min_bounds[0], y=y + min_bounds[1], z=z + min_bounds[2]
            ),
            type=block_type,
            orientation=orientation,
        )

    def create_block_list(
        self,
        arr,
        unique_int_to_block,
        block_priority=[],
        place_block_priority_first=True,
        block_orientation_dict={
            NORTH: [REDSTONE_BLOCK, PISTON, SLIME, QUARTZ_BLOCK],
            SOUTH: [STICKY_PISTON],
        },
        initial=None,
    ):
        unique_int_to_block = {
            int(k): unique_int_to_block[k] for k in unique_int_to_block
        }
        block_list = []
        if initial is None:
            initial = np.zeros(arr.shape)
        for index in np.ndindex(arr.shape):
            if (
                arr[index[0]][index[1]][index[2]]
                - initial[index[0]][index[1]][index[2]]
                != 0
            ):
                block_list.append(
                    self.create_block(
                        index,
                        unique_int_to_block[arr[index[0]][index[1]][index[2]]],
                        block_orientation_dict=block_orientation_dict,
                    )
                )
        unique_block_types = set([b.type for b in block_list])
        if len(block_priority) > 0:
            block_priority = [b for b in block_priority if b in unique_block_types]
            for b in block_priority:
                unique_block_types.remove(b)
            unique_block_types = list(unique_block_types)
            for b in block_priority:
                if not place_block_priority_first:
                    unique_block_types.append(b)
                else:
                    unique_block_types.insert(0, b)
            block_list = sorted(
                block_list, key=lambda x: unique_block_types.index(x.type)
            )
        blocks = Blocks(blocks=block_list)
        return blocks

    def spawn_steps(
        self,
        steps,
        unique_int_to_block,
        initial=None,
        out=None,
        take_argmax=True,
        block_priority=[REDSTONE_BLOCK],
        place_block_priority_first=True,
        block_orientation_dict={
            NORTH: [REDSTONE_BLOCK, PISTON, SLIME, QUARTZ_BLOCK],
            SOUTH: [STICKY_PISTON],
        },
        no_spawn=False,
        clear_blocks=True,
    ):
        unique_int_to_block = {
            int(k): unique_int_to_block[k] for k in unique_int_to_block
        }
        initial_blocks = None
        if clear_blocks:
            Blockloader.clear_blocks(
                self.client,
                (self.min_bounds[0] - 10, self.min_bounds[1], self.min_bounds[2] - 10),
                (
                    self.max_bounds[0] + 10,
                    self.max_bounds[1] + 10,
                    self.max_bounds[2] + 10,
                ),
            )
        if out is None:
            with torch.no_grad():
                out, out_arr, _ = self.trainer.rollout(initial=initial, steps=steps)
        blocks = out
        if take_argmax:
            blocks = rearrange(out, "b d h w c -> b w h d c").detach().cpu().numpy()[0]
            blocks = np.argmax(blocks[:, :, :, : self.trainer.num_categories], -1)
        else:
            blocks = rearrange(out, "w d h -> w h d")
        if initial is not None:
            initial_blocks = (
                rearrange(initial, "b d h w c -> b w h d c").detach().cpu().numpy()[0]
            )
            initial_blocks = np.argmax(
                initial_blocks[:, :, :, : self.trainer.num_categories], -1
            )
        block_list = self.create_block_list(
            blocks,
            unique_int_to_block,
            block_priority,
            place_block_priority_first,
            block_orientation_dict=block_orientation_dict,
            initial=initial_blocks,
        )
        if not no_spawn:
            self.client.spawnBlocks(block_list)
        return block_list, out

    def spawn(
        self,
        steps,
        initial_states=[None],
        block_priority=[REDSTONE_BLOCK],
        increments=5,
        wait=2,
    ):
        spawn_entities(
            [self],
            steps,
            self.trainer.dataset.target_unique_val_dict,
            initial_states=initial_states,
            block_priority=block_priority,
            increments=increments,
            wait=wait,
        )

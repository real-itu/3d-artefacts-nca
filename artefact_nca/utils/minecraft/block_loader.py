import copy
import math
import os
from os import listdir
from os.path import isfile, join

import grpc
import numpy as np
from matplotlib.colors import rgb2hex
from test_evocraft_py.minecraft_pb2 import *
from test_evocraft_py.minecraft_pb2_grpc import *

from artefact_nca.utils.minecraft.block_utils import BlockBuffer


class Blockloader:
    @staticmethod
    def spawn_nbt_blocks(
        dataset_dir: str,
        filename: str = "Extra_dark_oak.nbt",
        load_coord=(0, 10, 0),
        block_priority=[],
        place_block_priority_first=True,
    ) -> None:
        nbt_filenames = [
            join(dataset_dir, f)
            for f in listdir(dataset_dir)
            if isfile(join(dataset_dir, f)) and f.endswith("nbt")
        ]
        if filename is not None:
            nbt_filenames = [
                f for f in nbt_filenames if f == join(dataset_dir, filename)
            ]
        block_buffer = BlockBuffer()
        for f in nbt_filenames:
            block_buffer.send_nbt_to_server(
                load_coord,
                f,
                block_priority=block_priority,
                place_block_priority_first=place_block_priority_first,
            )
            load_coord = [load_coord[0] + 30, load_coord[1], load_coord[2]]

    @staticmethod
    def clear_blocks(client, min_coords, max_coords):

        client.fillCube(
            FillCubeRequest(  # Clear a 20x10x20 working area
                cube=Cube(
                    min=Point(x=min_coords[0], y=min_coords[1], z=min_coords[2]),
                    max=Point(x=max_coords[0], y=max_coords[1], z=max_coords[2],),
                ),
                type=5,
            )
        )

    @staticmethod
    def read_blocks(client, min_coords, max_coords):
        blocks = client.readCube(
            Cube(
                min=Point(x=min_coords[0], y=min_coords[1], z=min_coords[2]),
                max=Point(x=max_coords[0], y=max_coords[1], z=max_coords[2]),
            )
        )
        return blocks


def convert_to_color(arr, color_dict):
    new_arr = copy.deepcopy(arr).astype(object)
    for k in color_dict:
        new_arr[new_arr == int(k)] = color_dict[k]
    return new_arr


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_color_dict(unique_vals):
    state = np.random.RandomState(0)
    color_arr = list(state.uniform(0, 1, (len(unique_vals), 3)))
    color_arr = [rgb2hex(color) for color in color_arr]
    color_arr = [None] + color_arr
    colors = color_arr[: len(unique_vals)]
    color_dict = {str(i): colors[i] for i in range(len(unique_vals))}
    return color_dict


def get_block_array(
    client,
    min_coords,
    max_coords,
    unequal_padding=False,
    padding=None,
    no_padding=False,
):
    blocks = Blockloader.read_blocks(client, min_coords, max_coords)
    unique_vals = sorted(list(set([b.type for b in blocks.blocks])))
    unique_vals.remove(5)
    unique_vals.insert(0, 5)
    color_dict = get_color_dict(unique_vals)
    unique_val_to_int_dict = {unique_vals[i]: i for i in range(len(unique_vals))}
    unique_val_dict = {i: unique_vals[i] for i in range(len(unique_vals))}

    min_coords_shifted = np.array(min_coords)
    max_coords_shifted = np.array(max_coords)

    size_arr = np.array(max_coords_shifted) - np.array(min_coords_shifted) + 1
    arr = np.empty(size_arr, dtype=object)

    for b in blocks.blocks:
        w = b.position.x - min_coords_shifted[0]
        d = b.position.z - min_coords_shifted[2]
        h = b.position.y - min_coords_shifted[1]
        arr[w, d, h] = unique_val_to_int_dict[b.type]
    a = np.argwhere(arr > 0)
    l = []
    max_val = 0
    for i in range(3):
        min_arg = np.min(a[:, i])
        max_arg = np.max(a[:, i])
        l.append((min_arg, max_arg))
        if max_arg > max_val:
            max_val = max_arg

    sub_set = arr[l[0][0] : l[0][1] + 1, l[1][0] : l[1][1] + 1, l[2][0] : l[2][1] + 1]

    max_val = np.max(sub_set.shape)
    max_val = roundup(max_val)
    differences = [max_val - sub_set.shape[i] for i in range(3)]
    if unequal_padding:
        differences = [roundup(sub_set.shape[i]) - sub_set.shape[i] for i in range(3)]

    if no_padding:
        padding = [(0, 0), (0, 0), (0, 0)]
    if padding is None:
        padding = []
        for i in range(len(differences)):
            d = differences[i]
            left_pad = 0
            right_pad = d
            if i != 2:
                left_pad = d // 2
                right_pad = d // 2
                if left_pad + right_pad + sub_set.shape[i] < max_val:
                    right_pad += 1
            padding.append((left_pad, right_pad))

    arr = np.pad(sub_set, padding)
    unique_val_to_int_dict = {
        str(k): unique_val_to_int_dict[k] for k in unique_val_to_int_dict
    }
    unique_val_dict = {str(k): unique_val_dict[k] for k in unique_val_dict}
    return blocks, unique_val_dict, arr, color_dict, unique_val_dict


def read_nbt_target(
    nbt_path,
    load_coord=(50, 10, 10),
    load_range=30,
    unequal_padding=False,
    padding=None,
    no_padding=False,
    block_priority=[],
    place_block_priority_first=True,
):
    print("Block priority: ", block_priority)
    nbt_dir, nbt_file = os.path.split(nbt_path)
    channel = grpc.insecure_channel("localhost:5001")
    client = MinecraftServiceStub(channel)
    min_coords = (load_coord[0] - load_range, load_coord[1], load_coord[2] - load_range)
    max_coords = (
        load_coord[0] + load_range,
        load_coord[1] + load_range * 2,
        load_coord[2] + load_range,
    )

    Blockloader.clear_blocks(client, min_coords, max_coords)
    Blockloader.spawn_nbt_blocks(
        nbt_dir,
        filename=nbt_file,
        load_coord=load_coord,
        block_priority=block_priority,
        place_block_priority_first=place_block_priority_first,
    )
    return get_block_array(
        client, min_coords, max_coords, unequal_padding, padding, no_padding
    )


def create_flying_machine(load_coord=(50, 10, 10)):
    channel = grpc.insecure_channel("localhost:5001")
    client = MinecraftServiceStub(channel)
    min_coords = (load_coord[0] - 30, load_coord[1], load_coord[2] - 30)
    max_coords = (load_coord[0] + 30, load_coord[1] + 60, load_coord[2] + 30)

    Blockloader.clear_blocks(client, min_coords, max_coords)

    client.spawnBlocks(
        Blocks(
            blocks=[  # Spawn a flying machine
                # Lower layer
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] + 1
                    ),
                    type=PISTON,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2]
                    ),
                    type=SLIME,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] - 1
                    ),
                    type=STICKY_PISTON,
                    orientation=SOUTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] - 2
                    ),
                    type=PISTON,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 5, z=load_coord[2] - 4
                    ),
                    type=SLIME,
                    orientation=NORTH,
                ),
                # Upper layer
                # Activate
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 6, z=load_coord[2] - 1
                    ),
                    type=QUARTZ_BLOCK,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 6, z=load_coord[2]
                    ),
                    type=REDSTONE_BLOCK,
                    orientation=NORTH,
                ),
                Block(
                    position=Point(
                        x=load_coord[0] + 1, y=load_coord[1] + 6, z=load_coord[2] - 4
                    ),
                    type=REDSTONE_BLOCK,
                    orientation=NORTH,
                ),
            ]
        )
    )
    return get_block_array(client, min_coords, max_coords, False, None)

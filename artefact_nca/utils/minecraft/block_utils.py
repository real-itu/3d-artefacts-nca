import json

import grpc
import nbtlib
import test_evocraft_py.minecraft_pb2 as mcpb2
import test_evocraft_py.minecraft_pb2_grpc as mcraft_grpc
from test_evocraft_py.minecraft_pb2 import *

BLOCKS = BlockType.values()
ORIENTATIONS = Orientation.values()

block_directions = {
    "north": NORTH,
    "west": WEST,
    "south": SOUTH,
    "east": EAST,
    "up": UP,
    "down": DOWN,
}

block_direction_codes = lambda direction: block_directions[direction]


def move_coordinate(coord: (int, int, int), side_idx, delta=1):
    """A quick way to increment a coordinate in the desired direction"""
    switcher = [
        lambda c: (c[0], c[1], c[2] - delta),  # Go North
        lambda c: (c[0] - delta, c[1], c[2]),  # Go West
        lambda c: (c[0], c[1], c[2] + delta),  # Go South
        lambda c: (c[0] + delta, c[1], c[2]),  # Go East
        lambda c: (c[0], c[1] + delta, c[2]),  # Go Up
        lambda c: (c[0], c[1] - delta, c[2]),  # Go Down
    ]
    return switcher[side_idx](coord)


def blockname_to_blockid(blockname: str):
    try:
        return globals()[blockname.split(":")[1].upper()]
    except KeyError as e:
        print(f"unsupported block type {e}, putting STONE instead")
        return STONE


def dig_direction_from_facing_entry(pallete_entry: dict):
    if "Properties" in pallete_entry and "facing" in pallete_entry["Properties"]:
        return globals()[pallete_entry["Properties"]["facing"].upper()]
    else:
        return NORTH


class BlockBuffer:
    def __init__(self):
        self._blocks = []
        self._channel = grpc.insecure_channel("localhost:5001")
        self._client = mcraft_grpc.MinecraftServiceStub(self._channel)

    def add_block(
        self,
        coordinate: (int, int, int),
        orientation: Orientation,
        block_type: BlockType,
    ):
        assert block_type in BlockType.values(), "Unknown block type"
        assert orientation in Orientation.values(), "Unknown orientation"

        self._blocks.append(
            Block(
                position=Point(x=coordinate[0], y=coordinate[1], z=coordinate[2]),
                type=block_type,
                orientation=orientation,
            )
        )

    def save_block(
        self, start_coord: (int, int, int), end_coord: (int, int, int), save_file: str
    ):
        block_info = self.get_cube_info(start_coord, end_coord)

        block_info = [
            {
                "type": b.type,
                "orientation": b.orientation,
                "position": {
                    "x": b.position.x - start_coord[0],
                    "z": b.position.z - start_coord[2],
                    "y": b.position.y - start_coord[1],
                },
            }
            for b in block_info
            if b.type != AIR
        ]
        with open(save_file, "w") as handle:
            json.dump(block_info, handle)

    def send_json_to_server(self, load_coord: (int, int, int), load_file: str):
        """
        blocks json has the following format:
        [{"type": int, "position": {"x": int, "y": int, "z": int}, "orientation": int}, ...]
        """
        with open(load_file, "rb") as handle:
            blocks_from_json = json.load(handle)
            sx, sy, sz = load_coord
            block_list = [
                Block(
                    position=Point(
                        x=b["position"]["x"] + sx,
                        y=b["position"]["y"] + sy,
                        z=b["position"]["z"] + sz,
                    ),
                    type=b["type"],
                    orientation=b["orientation"],
                )
                for b in blocks_from_json
            ]
            response = self._client.spawnBlocks(Blocks(blocks=block_list))
            return response

    def send_nbt_to_server(
        self,
        load_coord: (int, int, int),
        load_file: str,
        block_priority=[],
        place_block_priority_first=True,
    ):
        """
        blocks json has the following format:
        [{"type": int, "position": {"x": int, "y": int, "z": int}, "orientation": int}, ...]
        """
        nbt_struct = nbtlib.load(load_file)
        nbt_blocks = nbt_struct.root["blocks"]
        nbt_palette = nbt_struct.root["palette"]

        blockname_to_blockid(nbt_palette[int(nbt_blocks[0]["state"])]["Name"])

        sx, sy, sz = load_coord
        block_list = [
            Block(
                position=Point(
                    x=int(b["pos"][0]) + sx,
                    y=int(b["pos"][1]) + sy,
                    z=int(b["pos"][2]) + sz,
                ),
                type=blockname_to_blockid(nbt_palette[int(b["state"])]["Name"]),
                orientation=dig_direction_from_facing_entry(
                    nbt_palette[int(b["state"])]
                ),
            )
            for b in nbt_blocks
        ]
        if len(block_priority) > 0:
            unique_block_types = set([b.type for b in block_list])
            block_priority = [b for b in block_priority if b in unique_block_types]
            for b in block_priority:
                unique_block_types.remove(b)
            unique_block_types = list(unique_block_types)
            if place_block_priority_first:
                for b in block_priority:
                    unique_block_types.insert(0, b)
            else:
                for b in block_priority:
                    unique_block_types.append(b)
            block_list = sorted(
                block_list, key=lambda x: unique_block_types.index(x.type)
            )
        response = self._client.spawnBlocks(Blocks(blocks=block_list))
        return response

    def send_to_server(self):
        response = self._client.spawnBlocks(Blocks(blocks=self._blocks))
        self._blocks = []
        return response

    def fill_cube(
        self,
        start_cord: (int, int, int),
        end_coord: (int, int, int),
        block_type: BlockType,
    ):
        assert block_type in BlockType.values(), "Unknown block type"

        min_x, max_x = (
            (start_cord[0], end_coord[0])
            if start_cord[0] < end_coord[0]
            else (end_coord[0], start_cord[0])
        )
        min_y, max_y = (
            (start_cord[1], end_coord[1])
            if start_cord[1] < end_coord[1]
            else (end_coord[1], start_cord[1])
        )
        min_z, max_z = (
            (start_cord[2], end_coord[2])
            if start_cord[2] < end_coord[2]
            else (end_coord[2], start_cord[2])
        )

        self._client.fillCube(
            FillCubeRequest(
                cube=Cube(
                    min=Point(x=min_x, y=min_y, z=min_z),
                    max=Point(x=max_x, y=max_y, z=max_z),
                ),
                type=block_type,
            )
        )

    def get_cube_info(self, start_cord: (int, int, int), end_coord: (int, int, int)):
        min_x, max_x = (
            (start_cord[0], end_coord[0])
            if start_cord[0] < end_coord[0]
            else (end_coord[0], start_cord[0])
        )
        min_y, max_y = (
            (start_cord[1], end_coord[1])
            if start_cord[1] < end_coord[1]
            else (end_coord[1], start_cord[1])
        )
        min_z, max_z = (
            (start_cord[2], end_coord[2])
            if start_cord[2] < end_coord[2]
            else (end_coord[2], start_cord[2])
        )

        response = self._client.readCube(
            Cube(
                min=Point(x=min_x, y=min_y, z=min_z),
                max=Point(x=max_x, y=max_y, z=max_z),
            )
        )

        return response.blocks

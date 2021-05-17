import abc
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class BaseTorchDataset(Dataset, metaclass=abc.ABCMeta):

    _config_group_ = "trainer/dataset"
    _config_name_ = "default"

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Random sample from dataset

        Args:
            batch_size (int): batch size to sample

        Returns:
            typing.Dict[str, typing.Any]: dict of outputs, ex:
            {"data": subset, "targets":targets}
        """
        max_indices = self.__len__()
        indices = torch.randint(0, max_indices, (batch_size,))
        return self.__getitem__(indices)

    def to_device(self, device: torch.device) -> None:
        pass

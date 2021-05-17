import torch
from torchsummaryX import summary


class BaseTorchModel(torch.nn.Module):
    def __init__(self):
        super(BaseTorchModel, self).__init__()

    def summary(self, input_shape):
        summary(self, input_shape)

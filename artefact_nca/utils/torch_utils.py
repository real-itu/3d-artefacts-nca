from typing import List

import torch


def create_linear_layer(
    input_dims: int, output_dims: int, bias: bool = True, activation=None
):
    linear = torch.nn.Linear(input_dims, output_dims, bias=bias)
    if activation is not None:
        linear = torch.nn.Sequential(*[linear, activation])
    return linear


def create_linear_network(
    input_dims: int, hidden_dims: List[int], output_dims: int, output_activation=None
):
    if len(hidden_dims) > 0:
        input_layer = create_linear_layer(
            input_dims, hidden_dims[0], bias=True, activation=torch.nn.ReLU()
        )
        hidden_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                hidden_layers.append(
                    create_linear_layer(
                        hidden_dims[i],
                        hidden_dims[i],
                        bias=True,
                        activation=torch.nn.ReLU(),
                    )
                )
            else:
                hidden_layers.append(
                    create_linear_layer(
                        hidden_dims[i - 1],
                        hidden_dims[i],
                        bias=True,
                        activation=torch.nn.ReLU(),
                    )
                )
        output_layer = create_linear_layer(
            hidden_dims[-1], output_dims, bias=True, activation=output_activation
        )
        return torch.nn.Sequential(*[input_layer, *hidden_layers, output_layer])
    return create_linear_layer(
        input_dims, output_dims, bias=True, activation=output_activation
    )

from typing import Sequence

import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
) -> nn.Sequential:
    layers = []
    prev_size = input_dim

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        prev_size = hidden_size

    layers.append(nn.Linear(prev_size, output_dim))
    return nn.Sequential(*layers)

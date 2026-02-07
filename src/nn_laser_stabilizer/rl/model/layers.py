from typing import Sequence

import torch
import torch.nn as nn

from nn_laser_stabilizer.rl.envs.box import Box


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


class Scaler(nn.Module):
    def __init__(self, action_space: Box):
        super(Scaler, self).__init__()
        self.low = action_space.low
        self.high = action_space.high
        self.tanh = nn.Tanh()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tanh = self.tanh(tensor)
        return self.low + (tanh + 1) * (self.high - self.low) / 2
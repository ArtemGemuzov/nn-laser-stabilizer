from typing import cast

import torch
import torch.nn as nn

from nn_laser_stabilizer.rl.envs.spaces.box import Box


class Scaler(nn.Module):
    def __init__(self, action_space: Box):
        super().__init__()
        self.register_buffer('low', action_space.low.clone())
        self.register_buffer('high', action_space.high.clone())

        range = action_space.high - action_space.low
        self.register_buffer('range', range.clone())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        low = cast(torch.Tensor, self.low)
        range = cast(torch.Tensor, self.range)

        tanh = torch.tanh(tensor)
        return low + (tanh + 1) * range / 2
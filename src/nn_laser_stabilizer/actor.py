from typing import Sequence

import torch

from nn_laser_stabilizer.layers import build_mlp
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.layers import Scaler


class Actor(Policy):
    pass


class MLPActor(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space: Box,
        hidden_sizes: Sequence[int] = (256, 256)
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, action_space=action_space, hidden_sizes=hidden_sizes)
        self.net_body = build_mlp(obs_dim, action_dim, hidden_sizes)
        self.scaler = Scaler(action_space)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.scaler(self.net_body(obs))


from typing import Sequence, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

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
    
    def forward(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        action = self.scaler(self.net_body(observation))
        return action, {}



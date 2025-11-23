from abc import ABC, abstractmethod

from typing import Sequence

import torch

from nn_laser_stabilizer.model import Model
from nn_laser_stabilizer.layers import build_mlp


class Critic(Model, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def value(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self(observation, action)
    

class MLPCritic(Critic):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256)
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.net = build_mlp(
            obs_dim + action_dim,
            1,
            hidden_sizes,
        )
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, action], dim=-1))
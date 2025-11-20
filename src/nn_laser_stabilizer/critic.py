from abc import ABC, abstractmethod

from typing import Sequence

import torch
import torch.nn as nn

from nn_laser_stabilizer.utils import build_mlp


class Critic(nn.Module, ABC):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
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
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__(obs_dim, action_dim)
        self.net = build_mlp(
            obs_dim + action_dim,
            1,
            hidden_sizes,
        )
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, action], dim=-1))
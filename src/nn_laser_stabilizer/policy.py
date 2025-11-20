from abc import ABC, abstractmethod

from typing import Sequence

import torch
import torch.nn as nn

from nn_laser_stabilizer.utils import build_mlp, Scaler


class Policy(nn.Module, ABC):   
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        pass
    
    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return self(observation)
    

class MLPPolicy(Policy):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__(obs_dim, action_dim)
        self.net_body = build_mlp(obs_dim, action_dim, hidden_sizes)
        self.scaler = Scaler(low=action_space.low, high=action_space.high)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.scaler(self.net_body(obs))


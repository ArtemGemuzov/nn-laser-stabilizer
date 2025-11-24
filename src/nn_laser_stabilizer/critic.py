from abc import ABC, abstractmethod

from typing import Sequence, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from nn_laser_stabilizer.model import Model
from nn_laser_stabilizer.layers import build_mlp


class Critic(Model, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    @torch.no_grad()
    def value(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self(observation, action, options)
    

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
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        q_value = self.net(torch.cat([observation, action], dim=-1))
        return q_value, {}

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
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


class RandomExplorationPolicy(Policy):
    def __init__(
        self,
        policy: Policy,
        exploration_steps: int,
        action_space,
    ):
        super().__init__(policy.obs_dim, policy.action_dim)
        self.policy = policy
        self.exploration_steps = exploration_steps
        
        self.action_low = torch.tensor(action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32)
        
        self._exploration_step_count = 0
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self._exploration_step_count < self.exploration_steps:
            action = torch.clamp(
                torch.randn(self.action_dim, dtype=torch.float32),
                self.action_low,
                self.action_high
            )
            self._exploration_step_count += 1
            return action
        else:
            return self.policy(observation)
    
    def train(self, mode: bool = True) -> nn.Module:
        self.policy.train(mode)
        return self
    
    def eval(self) -> nn.Module:
        self.policy.eval()
        return self
    
    def state_dict(self):
        return self.policy.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.policy.load_state_dict(state_dict)
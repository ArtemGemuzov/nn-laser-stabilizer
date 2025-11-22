from abc import ABC, abstractmethod

from typing import Sequence

import torch
import torch.nn as nn

from nn_laser_stabilizer.model import Model
from nn_laser_stabilizer.space import Box
from nn_laser_stabilizer.utils import build_mlp, Scaler


class Policy(Model, ABC):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
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
        action_space: Box,
        hidden_sizes: Sequence[int] = (256, 256)
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, action_space=action_space, hidden_sizes=hidden_sizes)
        self.net_body = build_mlp(obs_dim, action_dim, hidden_sizes)
        self.scaler = Scaler(action_space)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.scaler(self.net_body(obs))


class RandomExplorationPolicy(Policy):
    def __init__(
        self,
        policy: Policy,
        exploration_steps: int,
        action_space: Box
    ):
        super().__init__(
            obs_dim=policy._init_kwargs['obs_dim'], 
            action_dim=policy._init_kwargs['action_dim'],
            policy=policy,
            exploration_steps=exploration_steps,
            action_space=action_space
        )
        self.policy = policy
        self.exploration_steps = exploration_steps
        self.action_space = action_space
        
        self._exploration_step_count = 0
    
    def clone(self, reinitialize_weights: bool = False) -> "RandomExplorationPolicy":
        cloned_policy = self.policy.clone(reinitialize_weights=reinitialize_weights)
        
        new_model = self.__class__(
            policy=cloned_policy,
            exploration_steps=self.exploration_steps,
            action_space=self.action_space
        )
        
        return new_model
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1
            return self.action_space.sample()
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
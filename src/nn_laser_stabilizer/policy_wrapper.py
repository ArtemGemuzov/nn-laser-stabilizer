from typing import Tuple, Any, Optional, Dict

import torch
import torch.nn as nn

from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.actor import Actor


class RandomExplorationPolicy(Policy):
    def __init__(
        self,
        actor: Actor,
        exploration_steps: int,
        action_space: Box
    ):
        super().__init__(
            obs_dim=actor._init_kwargs['obs_dim'], 
            action_dim=actor._init_kwargs['action_dim'],
            actor=actor,
            exploration_steps=exploration_steps,
            action_space=action_space
        )
        self.actor = actor
        self.exploration_steps = exploration_steps
        self.action_space = action_space
        
        self._exploration_step_count = 0
    
    def clone(self, reinitialize_weights: bool = False) -> "RandomExplorationPolicy":
        cloned_actor = self.actor.clone(reinitialize_weights=reinitialize_weights)
        
        new_model = self.__class__(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
            action_space=self.action_space
        )
        
        return new_model
    
    def forward(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1
            action = self.action_space.sample()
            return action, {}
        else:
            return self.actor(observation, options)
    
    def train(self, mode: bool = True) -> nn.Module:
        self.actor.train(mode)
        return self
    
    def eval(self) -> nn.Module:
        self.actor.eval()
        return self
    
    def state_dict(self):
        return self.actor.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.actor.load_state_dict(state_dict)
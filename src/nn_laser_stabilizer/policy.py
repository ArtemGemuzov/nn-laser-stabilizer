from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict

import torch

from nn_laser_stabilizer.actor import Actor
from nn_laser_stabilizer.box import Box


class Policy(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def clone(self, reinitialize_weights: bool = False) -> "Policy":
        pass
    
    @abstractmethod
    def share_memory(self) -> "Policy":
        pass
    
    @abstractmethod
    def state_dict(self):
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    
    @abstractmethod
    def eval(self) -> "Policy":
        pass
    
    @abstractmethod
    def train(self, mode: bool = True) -> "Policy":
        pass


class DeterministicPolicy(Policy):
    def __init__(self, actor: Actor):
        self._actor = actor
    
    def act(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self._actor.act(observation, options)
    
    def clone(self, reinitialize_weights: bool = False) -> "DeterministicPolicy":
        cloned_actor = self._actor.clone(reinitialize_weights=reinitialize_weights)
        return DeterministicPolicy(actor=cloned_actor)
    
    def share_memory(self) -> "DeterministicPolicy":
        self._actor.share_memory()
        return self
    
    def state_dict(self):
        return self._actor.state_dict()
    
    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)
    
    def train(self, mode: bool = True) -> "DeterministicPolicy":
        self._actor.train(mode)
        return self
    
    def eval(self) -> "DeterministicPolicy":
        self._actor.eval()
        return self


class RandomExplorationPolicy(Policy):
    def __init__(
        self,
        actor: Actor,
        exploration_steps: int,
        action_space: Box
    ):
        self._actor = actor
        self.exploration_steps = exploration_steps
        self.action_space = action_space
        
        self._exploration_step_count = 0
    
    def act(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1
            action = self.action_space.sample()
            return action, {}
        else:
            return self._actor.act(observation, options)
    
    def clone(self, reinitialize_weights: bool = False) -> "RandomExplorationPolicy":
        cloned_actor = self._actor.clone(reinitialize_weights=reinitialize_weights)
        return RandomExplorationPolicy(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
            action_space=self.action_space
        )
    
    def share_memory(self) -> "RandomExplorationPolicy":
        self._actor.share_memory()
        return self
    
    def state_dict(self):
        return self._actor.state_dict()
    
    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)
    
    def train(self, mode: bool = True) -> "RandomExplorationPolicy":
        self._actor.train(mode)
        return self
    
    def eval(self) -> "RandomExplorationPolicy":
        self._actor.eval()
        return self


def make_policy(
    actor: Actor,
    action_space: Box,
    exploration_steps: int = 0,
) -> Policy:
    if exploration_steps > 0:
        return RandomExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            action_space=action_space
        )
    return DeterministicPolicy(actor=actor)
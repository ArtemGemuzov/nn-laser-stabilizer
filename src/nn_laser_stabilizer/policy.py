from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from nn_laser_stabilizer.actor import Actor
from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import ExplorationType


class Policy(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        ...
    
    @abstractmethod
    def clone(self) -> "Policy":
        ...
    
    @abstractmethod
    def share_memory(self) -> "Policy":
        ...
    
    @abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        ...
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        ...
    
    @abstractmethod
    def eval(self) -> "Policy":
        ...
    
    @abstractmethod
    def train(self, mode: bool = True) -> "Policy":
        ...
    
    @abstractmethod
    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        """Run light inference-only warmup without affecting exploration counters."""
        ...


class DeterministicPolicy(Policy):
    def __init__(self, actor: Actor):
        self._actor = actor
    
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._actor.act(observation, options)
    
    def clone(self) -> "DeterministicPolicy":
        cloned_actor = self._actor.clone()
        return DeterministicPolicy(actor=cloned_actor)
    
    def share_memory(self) -> "DeterministicPolicy":
        self._actor.share_memory()
        return self
    
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()
    
    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)
    
    def train(self, mode: bool = True) -> "DeterministicPolicy":
        self._actor.train(mode)
        return self
    
    def eval(self) -> "DeterministicPolicy":
        self._actor.eval()
        return self
    
    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._actor.eval()
        for _ in range(num_steps):
            fake_obs = observation_space.sample()
            self._actor.act(fake_obs, {})


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
    
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1
            action = self.action_space.sample()
            return action, {}
        else:
            return self._actor.act(observation, options)
    
    def clone(self) -> "RandomExplorationPolicy":
        cloned_actor = self._actor.clone()
        return RandomExplorationPolicy(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
            action_space=self.action_space
        )
    
    def share_memory(self) -> "RandomExplorationPolicy":
        self._actor.share_memory()
        return self
    
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()
    
    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)
    
    def train(self, mode: bool = True) -> "RandomExplorationPolicy":
        self._actor.train(mode)
        return self
    
    def eval(self) -> "RandomExplorationPolicy":
        self._actor.eval()
        return self
    
    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._actor.eval()
        for _ in range(num_steps):
            fake_obs = observation_space.sample()
            self._actor.act(fake_obs, {})


def make_policy(
    actor: Actor,
    action_space: Box,
    exploration_type: ExplorationType,
    exploration_steps: int,
) -> Policy:
    if exploration_type == ExplorationType.NONE:
        if exploration_steps != 0:
            raise ValueError(
                f"exploration_steps must be 0 when exploration_type is {ExplorationType.NONE}, "
                f"got exploration_steps={exploration_steps}"
            )
        return DeterministicPolicy(actor=actor)
    
    if exploration_type == ExplorationType.RANDOM:
        return RandomExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            action_space=action_space
        )
    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")


def make_policy_from_config(
    actor: Actor,
    action_space: Box,
    exploration_config: Config,
) -> Policy:
    return make_policy(
        actor=actor,
        action_space=action_space,
        exploration_type= ExplorationType.from_str(exploration_config.type),
        exploration_steps=exploration_config.steps,
    )
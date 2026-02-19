from abc import ABC, abstractmethod
from typing import Any

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy


class BaseExplorationPolicy(Policy, ABC):
    def __init__(self, inner: Policy, action_space: Box, exploration_steps: int):
        self._inner = inner
        self._action_space = action_space
        self._exploration_steps = exploration_steps
        self._exploration_step_count = 0

    @abstractmethod
    def _explore(self, action: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        ...

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        action, options = self._inner.act(observation, options)
        if self._exploration_step_count < self._exploration_steps:
            self._exploration_step_count += 1
            action = self._explore(action, options)
        return action, options

    def clone(self) -> "BaseExplorationPolicy":
        raise NotImplementedError("Subclasses must implement clone()")

    def share_memory(self) -> "BaseExplorationPolicy":
        self._inner.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "BaseExplorationPolicy":
        self._inner.train(mode)
        return self

    def eval(self) -> "BaseExplorationPolicy":
        self._inner.eval()
        return self

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._inner.warmup(observation_space, num_steps)

from abc import ABC, abstractmethod
from typing import Any

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy


class BaseExplorationPolicy(Policy, ABC):
    def __init__(
        self,
        inner: Policy,
        action_space: Box,
        *,
        start_step: int = 0,
        end_step: int | None = None,
    ):
        self._inner = inner
        self._action_space = action_space
        self._start_step = start_step
        self._end_step = end_step
        self._exploration_step_count = 0
        self._training = True

    @abstractmethod
    def _explore(self, action: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        ...

    def _in_exploration_window(self, step_index: int) -> bool:
        if step_index < self._start_step:
            return False
        if self._end_step is None:
            return True
        return step_index < self._end_step

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        action, options = self._inner.act(observation, options)
        policy_info = dict(options.get("policy_info", {}))
        policy_info["exploration_type"] = self.__class__.__name__
        policy_info["exploration_applied"] = False

        if self._training:
            step_index = self._exploration_step_count
            self._exploration_step_count += 1
        else:
            step_index = self._exploration_step_count

        if self._training and self._in_exploration_window(step_index):
            action = self._explore(action, options)
            policy_info["exploration_applied"] = True
        options["policy_info"] = policy_info
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
        self._training = mode
        self._inner.train(mode)
        return self

    def eval(self) -> "BaseExplorationPolicy":
        return self.train(False)

    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        self._inner.warmup(observation_space, num_steps)

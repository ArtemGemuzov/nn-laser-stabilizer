from typing import Any

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploaration import BaseExplorationPolicy


class RandomExplorationPolicy(BaseExplorationPolicy):
    def __init__(self, inner: Policy, action_space: Box, exploration_steps: int):
        super().__init__(inner, action_space, exploration_steps)

    def _explore(self, action: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        return self._action_space.sample()

    def clone(self) -> "RandomExplorationPolicy":
        return RandomExplorationPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            exploration_steps=self._exploration_steps,
        )

    @classmethod
    def from_config(cls, exploration_config: Config, *, policy: Policy, action_space: Box) -> "RandomExplorationPolicy":
        steps = int(exploration_config.steps)
        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for random exploration")
        return cls(inner=policy, action_space=action_space, exploration_steps=steps)

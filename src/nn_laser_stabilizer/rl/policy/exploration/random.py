from typing import Any

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploaration import BaseExplorationPolicy


class RandomExplorationPolicy(BaseExplorationPolicy):
    def __init__(
        self,
        inner: Policy,
        action_space: Box,
        *,
        start_step: int,
        end_step: int | None,
    ):
        super().__init__(inner, action_space, start_step=start_step, end_step=end_step)

    def _explore(self, action: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        return self._action_space.sample()

    def clone(self) -> "RandomExplorationPolicy":
        return RandomExplorationPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            start_step=self._start_step,
            end_step=self._end_step,
        )

    @classmethod
    def from_config(cls, exploration_config: Config, *, policy: Policy, action_space: Box) -> "RandomExplorationPolicy":
        start_step = int(exploration_config.get("start_step", 0))
        steps = int(exploration_config.get("steps", 0))
        end_step_raw = exploration_config.get("end_step", None)
        end_step = None if end_step_raw is None else int(end_step_raw)
        if start_step < 0:
            raise ValueError("exploration.start_step must be >= 0 for random exploration")
        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for random exploration")
        if end_step is None:
            end_step = start_step + steps
        if end_step < start_step:
            raise ValueError("exploration.end_step must be >= exploration.start_step for random exploration")
        return cls(inner=policy, action_space=action_space, start_step=start_step, end_step=end_step)

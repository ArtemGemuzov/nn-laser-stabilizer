from typing import Any

import torch
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.envs.spaces.discrete import Discrete
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploration import BaseExplorationPolicy


class EpsilonGreedyPolicy(BaseExplorationPolicy):
    def __init__(
        self,
        inner: Policy,
        action_space: Discrete,
        *,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
    ):
        super().__init__(inner, action_space, start_step=0, end_step=None)
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps

    @property
    def _epsilon(self) -> float:
        t = min(self._exploration_step_count / max(self._epsilon_decay_steps, 1), 1.0)
        return self._epsilon_start + (self._epsilon_end - self._epsilon_start) * t

    def _explore(self, action: Tensor, options: dict[str, Any]) -> Tensor:
        if torch.rand(1).item() < self._epsilon:
            return torch.randint(0, self._action_space.n, (1,)).float()
        return action

    def clone(self) -> "EpsilonGreedyPolicy":
        return EpsilonGreedyPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            epsilon_start=self._epsilon_start,
            epsilon_end=self._epsilon_end,
            epsilon_decay_steps=self._epsilon_decay_steps,
        )

    @classmethod
    def from_config(
        cls,
        exploration_config: Config,
        *,
        policy: Policy,
        action_space: Box | Discrete,
    ) -> "EpsilonGreedyPolicy":
        if not isinstance(action_space, Discrete):
            raise ValueError("epsilon_greedy exploration requires a discrete action space")
        epsilon_start = float(exploration_config.get("epsilon_start", 1.0))
        epsilon_end = float(exploration_config.get("epsilon_end", 0.05))
        epsilon_decay_steps = int(exploration_config.get("epsilon_decay_steps", 10_000))
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1")
        if epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be > 0")
        return cls(
            inner=policy,
            action_space=action_space,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )

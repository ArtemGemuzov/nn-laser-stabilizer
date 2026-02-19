from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.model.actor import Actor
from nn_laser_stabilizer.rl.policy.policy import Policy

import torch

from typing import Any


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

    @classmethod
    def from_config(
        cls,
        exploration_config: Config,
        *,
        actor: Actor,
    ) -> "DeterministicPolicy":
        steps = int(exploration_config.steps)
        if steps != 0:
            raise ValueError(
                "exploration.steps must be 0 when using DeterministicPolicy (type=none)"
            )
        return cls(actor=actor)
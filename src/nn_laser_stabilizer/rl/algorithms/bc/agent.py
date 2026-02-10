from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.actor import Actor, make_actor_from_config
from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.utils import make_policy_from_config


class BCAgent(Agent):
    ACTOR_FILENAME = "actor.pth"

    def __init__(self, actor: Actor):
        self._actor = actor

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "BCAgent":
        actor_config = algorithm_config.actor

        actor = make_actor_from_config(
            network_config=actor_config.network,
            action_space=action_space,
            observation_space=observation_space,
        ).train()

        return cls(actor=actor)

    def forward_train(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
    ) -> dict[str, Any]:
        predicted_actions, _ = self._actor(observations)

        return {
            "actions": predicted_actions,
            "dataset_actions": actions,
        }

    @torch.no_grad()
    def forward_action(self, observation: Tensor) -> Tensor:
        action, _ = self._actor(observation)
        return action

    def policy(self, exploration_config: Config) -> Policy:
        return make_policy_from_config(
            actor=self._actor,
            exploration_config=exploration_config,
        )

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self._actor.save(models_dir / self.ACTOR_FILENAME)

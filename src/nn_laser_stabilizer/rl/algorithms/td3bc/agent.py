from typing import Any

import torch
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.model.actor import Actor, make_actor_from_config
from nn_laser_stabilizer.rl.model.critic import Critic, make_critic_from_config
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer
from nn_laser_stabilizer.rl.envs.box import Box


class TD3BCAgent(TD3Agent):
    EPSILON = 1e-8

    def forward_train(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
    ) -> dict[str, Any]:
        # Get base TD3 output
        base = dict(super().forward_train(
            observations, actions, rewards, next_observations, dones,
        ))

        # Additional outputs for BC term
        actor_actions, _ = self._actor(observations)

        with torch.no_grad():
            q_dataset, _ = self._critic1(observations, actions)
            lambda_coef = 1.0 / (torch.abs(q_dataset).mean().item() + self.EPSILON)

        base["actor_actions"] = actor_actions
        base["dataset_actions"] = actions
        base["lambda_coef"] = lambda_coef

        return base

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "TD3BCAgent":
        policy_noise = float(algorithm_config.policy_noise)
        noise_clip = float(algorithm_config.noise_clip)

        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

        actor = make_actor_from_config(
            network_config=actor_config.network,
            action_space=action_space,
            observation_space=observation_space,
        ).train()

        critic = make_critic_from_config(
            network_config=critic_config.network,
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
        ).train()

        return cls(
            actor=actor,
            critic=critic,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )

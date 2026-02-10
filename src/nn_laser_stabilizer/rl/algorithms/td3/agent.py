from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.actor import Actor, make_actor_from_config
from nn_laser_stabilizer.rl.model.critic import Critic, make_critic_from_config
from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.utils import make_policy_from_config


class TD3Agent(Agent):
    ACTOR_FILENAME = "actor.pth"
    CRITIC1_FILENAME = "critic1.pth"
    CRITIC2_FILENAME = "critic2.pth"
    ACTOR_TARGET_FILENAME = "actor_target.pth"
    CRITIC1_TARGET_FILENAME = "critic1_target.pth"
    CRITIC2_TARGET_FILENAME = "critic2_target.pth"

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        policy_noise: float,
        noise_clip: float,
    ):
        self._actor = actor
        self._critic1 = critic
        self._critic2 = critic.clone(reinitialize_weights=True)

        self._actor_target = self._actor.clone().requires_grad_(False)
        self._critic1_target = self._critic1.clone().requires_grad_(False)
        self._critic2_target = self._critic2.clone().requires_grad_(False)

        action_space = actor.action_space
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._min_action = action_space.low
        self._max_action = action_space.high

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "TD3Agent":
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

    # ------------------------------------------------------------------
    # forward_train / forward_action
    # ------------------------------------------------------------------

    def forward_train(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
    ) -> dict[str, Any]:
        # --- critic targets (no grad) ---
        with torch.no_grad():
            next_actions, _ = self._actor_target(next_observations)
            noise = (torch.randn_like(next_actions) * self._policy_noise).clamp(
                -self._noise_clip, self._noise_clip
            )
            next_actions = (next_actions + noise).clamp(self._min_action, self._max_action)

            target_q1, _ = self._critic1_target(next_observations, next_actions)
            target_q2, _ = self._critic2_target(next_observations, next_actions)

        # --- current critics ---
        current_q1, _ = self._critic1(observations, actions)
        current_q2, _ = self._critic2(observations, actions)

        # --- actor ---
        actor_actions, _ = self._actor(observations)
        actor_q_value, _ = self._critic1(observations, actor_actions)

        return {
            "current_q1": current_q1,
            "current_q2": current_q2,
            "target_q1": target_q1,
            "target_q2": target_q2,
            "rewards": rewards,
            "dones": dones,
            "actor_q_value": actor_q_value,
        }

    @torch.no_grad()
    def forward_action(self, observation: Tensor) -> Tensor:
        action, _ = self._actor(observation)
        return action

    # ------------------------------------------------------------------
    # policy / save
    # ------------------------------------------------------------------

    def policy(self, exploration_config: Config) -> Policy:
        return make_policy_from_config(
            actor=self._actor,
            exploration_config=exploration_config,
        )

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self._actor.save(models_dir / self.ACTOR_FILENAME)
        self._critic1.save(models_dir / self.CRITIC1_FILENAME)
        self._critic2.save(models_dir / self.CRITIC2_FILENAME)
        self._actor_target.save(models_dir / self.ACTOR_TARGET_FILENAME)
        self._critic1_target.save(models_dir / self.CRITIC1_TARGET_FILENAME)
        self._critic2_target.save(models_dir / self.CRITIC2_TARGET_FILENAME)

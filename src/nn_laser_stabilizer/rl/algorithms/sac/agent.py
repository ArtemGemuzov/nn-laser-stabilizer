from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import NetworkType
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.critic import Critic, make_critic_from_config
from nn_laser_stabilizer.rl.model.layers import build_mlp
from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.gaussian import (
    GaussianPolicy,
    tanh_squash,
    gaussian_log_prob,
    LOG_STD_MIN,
    LOG_STD_MAX,
)


class SACAgent(Agent):
    ACTOR_FILENAME = "sac_actor.pth"
    CRITIC1_FILENAME = "critic1.pth"
    CRITIC2_FILENAME = "critic2.pth"
    CRITIC1_TARGET_FILENAME = "critic1_target.pth"
    CRITIC2_TARGET_FILENAME = "critic2_target.pth"

    def __init__(
        self,
        actor_net: nn.Module,
        critic: Critic,
        action_space: Box,
    ):
        self._actor_net = actor_net
        self._critic1 = critic
        self._critic2 = critic.clone(reinitialize_weights=True)
        self._critic1_target = self._critic1.clone().requires_grad_(False)
        self._critic2_target = self._critic2.clone().requires_grad_(False)
        self._action_space = action_space

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "SACAgent":
        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

        # Build raw MLP with 2 * action_dim outputs (mean + log_std)
        network_config = actor_config.network
        network_type = NetworkType.from_str(network_config.type)
        if network_type != NetworkType.MLP:
            raise ValueError(f"SAC currently supports only MLP actor, got: {network_type}")

        hidden_sizes = tuple(int(h) for h in network_config.mlp_hidden_sizes)
        actor_net = build_mlp(
            input_dim=observation_space.dim,
            output_dim=2 * action_space.dim,
            hidden_sizes=hidden_sizes,
        ).train()

        critic = make_critic_from_config(
            network_config=critic_config.network,
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
        ).train()

        return cls(
            actor_net=actor_net,
            critic=critic,
            action_space=action_space,
        )

    # ------------------------------------------------------------------
    # Internal sampling helper
    # ------------------------------------------------------------------

    def _sample_actions(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Sample actions via reparameterization and return (actions, log_prob)."""
        raw = self._actor_net(observations)
        mean, log_std = raw.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        actions = tanh_squash(x_t, self._action_space.low, self._action_space.high)
        log_prob = gaussian_log_prob(normal, x_t)
        return actions, log_prob

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
        # --- current critics ---
        current_q1, _ = self._critic1(observations, actions)
        current_q2, _ = self._critic2(observations, actions)

        # --- target critics with sampled next actions (no grad) ---
        with torch.no_grad():
            next_actions, next_log_prob = self._sample_actions(next_observations)
            target_q1, _ = self._critic1_target(next_observations, next_actions)
            target_q2, _ = self._critic2_target(next_observations, next_actions)

        # --- actor (current) ---
        actor_actions, actor_log_prob = self._sample_actions(observations)
        actor_q1, _ = self._critic1(observations, actor_actions)
        actor_q2, _ = self._critic2(observations, actor_actions)

        return {
            "current_q1": current_q1,
            "current_q2": current_q2,
            "target_q1": target_q1,
            "target_q2": target_q2,
            "next_log_prob": next_log_prob,
            "rewards": rewards,
            "dones": dones,
            "actor_q1": actor_q1,
            "actor_q2": actor_q2,
            "actor_log_prob": actor_log_prob,
        }

    @torch.no_grad()
    def forward_action(self, observation: Tensor) -> Tensor:
        raw = self._actor_net(observation)
        mean, _log_std = raw.chunk(2, dim=-1)
        return tanh_squash(mean, self._action_space.low, self._action_space.high)

    # ------------------------------------------------------------------
    # policy / save
    # ------------------------------------------------------------------

    def policy(self, exploration_config: Config) -> Policy:
        return GaussianPolicy(net=self._actor_net, action_space=self._action_space)

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._actor_net.state_dict(), models_dir / self.ACTOR_FILENAME)
        self._critic1.save(models_dir / self.CRITIC1_FILENAME)
        self._critic2.save(models_dir / self.CRITIC2_FILENAME)
        self._critic1_target.save(models_dir / self.CRITIC1_TARGET_FILENAME)
        self._critic2_target.save(models_dir / self.CRITIC2_TARGET_FILENAME)

from pathlib import Path

import torch
import torch.nn as nn

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import NetworkType
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.model.critic import Critic, make_critic_from_config
from nn_laser_stabilizer.rl.model.layers import build_mlp
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.gaussian import GaussianPolicy


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
        self.actor_net = actor_net
        self.critic1 = critic
        self.critic2 = critic.clone(reinitialize_weights=True)
        self.critic1_target = self.critic1.clone().requires_grad_(False)
        self.critic2_target = self.critic2.clone().requires_grad_(False)
        self.action_space = action_space

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "SACAgent":
        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

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

    def exploration_policy(self, exploration_config: Config) -> Policy:
        return GaussianPolicy(net=self.actor_net, action_space=self.action_space)

    def default_policy(self) -> Policy:
        return GaussianPolicy(net=self.actor_net, action_space=self.action_space).eval()

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_net.state_dict(), models_dir / self.ACTOR_FILENAME)
        self.critic1.save(models_dir / self.CRITIC1_FILENAME)
        self.critic2.save(models_dir / self.CRITIC2_FILENAME)
        self.critic1_target.save(models_dir / self.CRITIC1_TARGET_FILENAME)
        self.critic2_target.save(models_dir / self.CRITIC2_TARGET_FILENAME)

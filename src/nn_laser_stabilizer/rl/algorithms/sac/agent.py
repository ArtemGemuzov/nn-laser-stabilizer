from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import NetworkType
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs
from nn_laser_stabilizer.rl.model.stochastic_actor import StochasticActor
from nn_laser_stabilizer.rl.model.critic import Critic
from nn_laser_stabilizer.rl.networks.factory import make_actor_network_from_config, make_critic_network_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.stochastic import StochasticPolicy
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config


class SACAgent(Agent):
    def __init__(
        self,
        actor: StochasticActor,
        critic1: Critic,
        critic2: Critic,
        critic1_target: Critic,
        critic2_target: Critic,
        action_space: Box,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        alpha_optimizer: Optimizer,
        soft_updater: SoftUpdater,
        log_alpha: nn.Parameter,
        target_entropy: float,
        gamma: float,
    ):
        self._actor = actor
        self._critic1 = critic1
        self._critic2 = critic2
        self._critic1_target = critic1_target
        self._critic2_target = critic2_target
        self._action_space = action_space

        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer
        self._soft_updater = soft_updater
        self._log_alpha = log_alpha
        self._target_entropy = target_entropy
        self._gamma = gamma

    @property
    def alpha(self) -> Tensor:
        return self._log_alpha.exp()

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

        actor_network = make_actor_network_from_config(
            network_config=network_config,
            obs_dim=observation_space.dim,
            output_dim=2 * action_space.dim,
        )
        actor = StochasticActor(network=actor_network, action_space=action_space).train()

        critic_network = make_critic_network_from_config(
            network_config=critic_config.network,
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
        )
        critic1 = Critic(network=critic_network).train()
        critic2 = critic1.clone(reinitialize_weights=True).train()
        critic1_target = critic1.clone().requires_grad_(False)
        critic2_target = critic2.clone().requires_grad_(False)

        gamma = float(algorithm_config.gamma)
        tau = float(algorithm_config.tau)
        initial_alpha = float(algorithm_config.initial_alpha)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")
        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")

        log_alpha = nn.Parameter(torch.tensor(initial_alpha).log())
        target_entropy = -float(action_space.dim)

        actor_optimizer = Optimizer(actor.parameters(), lr=float(actor_config.optimizer.lr))
        critic_optimizer = Optimizer(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=float(critic_config.optimizer.lr),
        )
        alpha_optimizer = Optimizer([log_alpha], lr=float(algorithm_config.alpha_optimizer.lr))

        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (critic1_target, critic1),
                    (critic2_target, critic2),
                )
            ),
            tau=tau,
        )

        return cls(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            critic1_target=critic1_target,
            critic2_target=critic2_target,
            action_space=action_space,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            soft_updater=soft_updater,
            log_alpha=log_alpha,
            target_entropy=target_entropy,
            gamma=gamma,
        )

    def exploration_policy(self, exploration_config: Config) -> Policy:
        base_policy = StochasticPolicy(actor=self._actor)
        return make_exploration_policy_from_config(
            policy=base_policy,
            action_space=self._action_space,
            exploration_config=exploration_config,
        )

    def default_policy(self) -> Policy:
        return StochasticPolicy(actor=self._actor).eval()

    def _critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        alpha = self.alpha.detach()

        current_q1 = self._critic1(obs, actions).q_value
        current_q2 = self._critic2(obs, actions).q_value

        with torch.no_grad():
            next_output = self._actor(next_obs)
            target_q1 = self._critic1_target(next_obs, next_output.action).q_value
            target_q2 = self._critic2_target(next_obs, next_output.action).q_value
            target_q = torch.min(target_q1, target_q2) - alpha * next_output.log_prob
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        return F.mse_loss(current_q1, target_q), F.mse_loss(current_q2, target_q)

    def _actor_loss(self, obs: Tensor) -> dict[str, Tensor]:
        alpha = self.alpha.detach()

        output = self._actor(obs)
        actor_q1 = self._critic1(obs, output.action).q_value
        actor_q2 = self._critic2(obs, output.action).q_value

        min_q = torch.min(actor_q1, actor_q2)
        actor_loss = (alpha * output.log_prob - min_q).mean()

        return {
            "actor_loss": actor_loss,
            "actor_log_prob": output.log_prob,
        }

    def _alpha_loss(self, actor_log_prob: Tensor) -> Tensor:
        return -(self._log_alpha * (actor_log_prob + self._target_entropy).detach()).mean()

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._critic_loss(obs, actions, rewards, next_obs, dones)
        self._critic_optimizer.step(loss_q1 + loss_q2)

        actor_result = self._actor_loss(obs)
        self._actor_optimizer.step(actor_result["actor_loss"])

        alpha_loss = self._alpha_loss(actor_result["actor_log_prob"])
        self._alpha_optimizer.step(alpha_loss)

        self._soft_updater.update()

        return {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "actor_loss": actor_result["actor_loss"].item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._actor.save(path / 'actor.pt')
        self._critic1.save(path / 'critic1.pt')
        self._critic2.save(path / 'critic2.pt')
        self._critic1_target.save(path / 'critic1_target.pt')
        self._critic2_target.save(path / 'critic2_target.pt')
        self._actor_optimizer.save(path / 'actor_optimizer.pt')
        self._critic_optimizer.save(path / 'critic_optimizer.pt')
        self._alpha_optimizer.save(path / 'alpha_optimizer.pt')
        torch.save(self._log_alpha.data, path / 'log_alpha.pt')

    def load(self, path: Path) -> None:
        path = Path(path)
        self._actor.load(path / 'actor.pt')
        self._critic1.load(path / 'critic1.pt')
        self._critic2.load(path / 'critic2.pt')
        self._critic1_target.load(path / 'critic1_target.pt')
        self._critic2_target.load(path / 'critic2_target.pt')
        self._actor_optimizer.load(path / 'actor_optimizer.pt')
        self._critic_optimizer.load(path / 'critic_optimizer.pt')
        self._alpha_optimizer.load(path / 'alpha_optimizer.pt')
        self._log_alpha.data = torch.load(path / 'log_alpha.pt', map_location='cpu', weights_only=True)

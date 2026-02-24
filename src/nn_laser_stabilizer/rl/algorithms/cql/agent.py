from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.model.critic import Critic
from nn_laser_stabilizer.rl.model.stochastic_actor import StochasticActor
from nn_laser_stabilizer.rl.networks.factory import (
    NetworkType,
    make_actor_network_from_config,
    make_critic_network_from_config,
)
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.stochastic import StochasticPolicy


class CQLAgent(Agent):
    DIR_NAME = "agent_cql"
    EPS = 1e-6

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
        cql_alpha: float,
        cql_temperature: float,
        cql_n_actions: int,
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

        self._cql_alpha = cql_alpha
        self._cql_temperature = cql_temperature
        self._cql_n_actions = cql_n_actions

    @property
    def alpha(self) -> Tensor:
        return self._log_alpha.exp()

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "CQLAgent":
        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

        network_config = actor_config.network
        network_type = NetworkType.from_str(network_config.type)
        if network_type != NetworkType.MLP:
            raise ValueError(f"CQL currently supports only MLP actor, got: {network_type}")

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
        cql_alpha = float(algorithm_config.cql_alpha)
        cql_temperature = float(algorithm_config.cql_temperature)
        cql_n_actions = int(algorithm_config.cql_n_actions)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")
        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")
        if cql_alpha < 0.0:
            raise ValueError("algorithm.cql_alpha must be >= 0")
        if cql_temperature <= 0.0:
            raise ValueError("algorithm.cql_temperature must be > 0")
        if cql_n_actions <= 0:
            raise ValueError("algorithm.cql_n_actions must be > 0")

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
            cql_alpha=cql_alpha,
            cql_temperature=cql_temperature,
            cql_n_actions=cql_n_actions,
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

    def _sample_uniform_actions(self, batch_size: int, n_actions: int, obs: Tensor) -> Tensor:
        low = self._action_space.low.to(device=obs.device, dtype=obs.dtype)
        high = self._action_space.high.to(device=obs.device, dtype=obs.dtype)
        random = torch.rand(batch_size, n_actions, self._action_space.dim, device=obs.device, dtype=obs.dtype)
        return low + random * (high - low)

    def _critic_values_for_actions(self, critic: Critic, obs: Tensor, actions: Tensor) -> Tensor:
        batch_size, n_actions, action_dim = actions.shape
        repeated_obs = obs.unsqueeze(1).expand(-1, n_actions, -1).reshape(batch_size * n_actions, -1)
        flat_actions = actions.reshape(batch_size * n_actions, action_dim)
        flat_q_values = critic(repeated_obs, flat_actions).q_value
        return flat_q_values.reshape(batch_size, n_actions, 1)

    def _sample_policy_actions(self, obs: Tensor, n_actions: int) -> tuple[Tensor, Tensor]:
        batch_size = obs.shape[0]
        repeated_obs = obs.unsqueeze(1).expand(-1, n_actions, -1).reshape(batch_size * n_actions, -1)
        actor_output = self._actor(repeated_obs)
        actions = actor_output.action.reshape(batch_size, n_actions, self._action_space.dim)
        log_probs = actor_output.log_prob.reshape(batch_size, n_actions, 1)
        return actions, log_probs

    def _conservative_loss(self, critic: Critic, obs: Tensor, next_obs: Tensor, data_actions: Tensor) -> Tensor:
        obs = obs.reshape(-1, obs.shape[-1])
        next_obs = next_obs.reshape(-1, next_obs.shape[-1])
        data_actions = data_actions.reshape(-1, data_actions.shape[-1])
        batch_size = obs.shape[0]

        random_actions = self._sample_uniform_actions(batch_size, self._cql_n_actions, obs)
        random_q_values = self._critic_values_for_actions(critic, obs, random_actions)

        with torch.no_grad():
            current_actions, current_log_probs = self._sample_policy_actions(obs, self._cql_n_actions)
            next_actions, next_log_probs = self._sample_policy_actions(next_obs, self._cql_n_actions)
        current_q_values = self._critic_values_for_actions(critic, obs, current_actions)
        next_q_values = self._critic_values_for_actions(critic, obs, next_actions)

        action_range = (self._action_space.high - self._action_space.low).to(
            device=obs.device,
            dtype=obs.dtype,
        )
        random_log_prob = -torch.log(action_range + self.EPS).sum()

        cql_logits = torch.cat(
            (
                random_q_values - random_log_prob,
                current_q_values - current_log_probs.detach(),
                next_q_values - next_log_probs.detach(),
            ),
            dim=1,
        )

        data_q_values = critic(obs, data_actions).q_value
        conservative_q = torch.logsumexp(cql_logits / self._cql_temperature, dim=1)
        conservative_q = conservative_q * self._cql_temperature
        return (conservative_q - data_q_values).mean()

    def _critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = self.alpha.detach()

        current_q1 = self._critic1(obs, actions).q_value
        current_q2 = self._critic2(obs, actions).q_value

        with torch.no_grad():
            next_output = self._actor(next_obs)
            target_q1 = self._critic1_target(next_obs, next_output.action).q_value
            target_q2 = self._critic2_target(next_obs, next_output.action).q_value
            target_q = torch.min(target_q1, target_q2) - alpha * next_output.log_prob
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        bellman_q1 = F.mse_loss(current_q1, target_q)
        bellman_q2 = F.mse_loss(current_q2, target_q)
        cql_q1 = self._conservative_loss(self._critic1, obs, next_obs, actions)
        cql_q2 = self._conservative_loss(self._critic2, obs, next_obs, actions)

        loss_q1 = bellman_q1 + self._cql_alpha * cql_q1
        loss_q2 = bellman_q2 + self._cql_alpha * cql_q2

        return loss_q1, loss_q2, cql_q1, cql_q2

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

        loss_q1, loss_q2, cql_q1, cql_q2 = self._critic_loss(obs, actions, rewards, next_obs, dones)
        self._critic_optimizer.step(loss_q1 + loss_q2)

        actor_result = self._actor_loss(obs)
        self._actor_optimizer.step(actor_result["actor_loss"])

        alpha_loss = self._alpha_loss(actor_result["actor_log_prob"])
        self._alpha_optimizer.step(alpha_loss)

        self._soft_updater.update()

        return {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "cql_q1": cql_q1.item(),
            "cql_q2": cql_q2.item(),
            "actor_loss": actor_result["actor_loss"].item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, path: Path | None = None) -> None:
        path = Path(path) if path is not None else self.default_path
        path.mkdir(parents=True, exist_ok=True)
        self._actor.save(path / "actor.pt")
        self._critic1.save(path / "critic1.pt")
        self._critic2.save(path / "critic2.pt")
        self._critic1_target.save(path / "critic1_target.pt")
        self._critic2_target.save(path / "critic2_target.pt")
        self._actor_optimizer.save(path / "actor_optimizer.pt")
        self._critic_optimizer.save(path / "critic_optimizer.pt")
        self._alpha_optimizer.save(path / "alpha_optimizer.pt")
        torch.save(self._log_alpha.data, path / "log_alpha.pt")

    def load(self, path: Path) -> None:
        path = Path(path)
        self._actor.load(path / "actor.pt")
        self._critic1.load(path / "critic1.pt")
        self._critic2.load(path / "critic2.pt")
        self._critic1_target.load(path / "critic1_target.pt")
        self._critic2_target.load(path / "critic2_target.pt")
        self._actor_optimizer.load(path / "actor_optimizer.pt")
        self._critic_optimizer.load(path / "critic_optimizer.pt")
        self._alpha_optimizer.load(path / "alpha_optimizer.pt")
        self._log_alpha.data = torch.load(path / "log_alpha.pt", map_location="cpu", weights_only=True)

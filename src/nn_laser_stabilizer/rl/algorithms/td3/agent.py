from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Agent
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs
from nn_laser_stabilizer.rl.model.deterministic_actor import DeterministicActor
from nn_laser_stabilizer.rl.model.critic import Critic
from nn_laser_stabilizer.rl.networks.factory import make_actor_network_from_config, make_critic_network_from_config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.deterministic import DeterministicPolicy
from nn_laser_stabilizer.rl.policy.factory import make_exploration_policy_from_config


class TD3Agent(Agent):
    DIR_NAME = "agent_td3"

    def __init__(
        self,
        actor: DeterministicActor,
        actor_target: DeterministicActor,
        critic1: Critic,
        critic2: Critic,
        critic1_target: Critic,
        critic2_target: Critic,
        action_space: Box,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        soft_updater: SoftUpdater,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
        policy_freq: int,
    ):
        self._actor = actor
        self._actor_target = actor_target
        self._critic1 = critic1
        self._critic2 = critic2
        self._critic1_target = critic1_target
        self._critic2_target = critic2_target
        self._action_space = action_space

        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._soft_updater = soft_updater
        self._gamma = gamma
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_freq = policy_freq
        self._step = 0

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "TD3Agent":
        actor_config = algorithm_config.actor
        critic_config = algorithm_config.critic

        actor_network = make_actor_network_from_config(
            network_config=actor_config.network,
            obs_dim=observation_space.dim,
            output_dim=action_space.dim,
        )
        actor = DeterministicActor(network=actor_network, action_space=action_space).train()
        actor_target = actor.clone().requires_grad_(False)

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
        policy_noise = float(algorithm_config.policy_noise)
        noise_clip = float(algorithm_config.noise_clip)
        tau = float(algorithm_config.tau)
        policy_freq = int(algorithm_config.policy_freq)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")
        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")
        if policy_freq <= 0:
            raise ValueError("algorithm.policy_freq must be > 0")

        actor_optimizer = Optimizer(actor.parameters(), lr=float(actor_config.optimizer.lr))
        critic_optimizer = Optimizer(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=float(critic_config.optimizer.lr),
        )
        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (actor_target, actor),
                    (critic1_target, critic1),
                    (critic2_target, critic2),
                )
            ),
            tau=tau,
        )

        return cls(
            actor=actor,
            actor_target=actor_target,
            critic1=critic1,
            critic2=critic2,
            critic1_target=critic1_target,
            critic2_target=critic2_target,
            action_space=action_space,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            soft_updater=soft_updater,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
        )

    def exploration_policy(self, exploration_config: Config) -> Policy:
        base_policy = DeterministicPolicy(actor=self._actor)
        return make_exploration_policy_from_config(
            policy=base_policy,
            action_space=self._action_space,
            exploration_config=exploration_config,
        )

    def default_policy(self) -> Policy:
        return DeterministicPolicy(actor=self._actor).eval()

    def _critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            target_output = self._actor_target(next_obs)
            noise = (torch.randn_like(target_output.action) * self._policy_noise).clamp(
                -self._noise_clip, self._noise_clip
            )
            next_actions = torch.clamp(
                target_output.action + noise,
                self._action_space.low,
                self._action_space.high,
            )

            target_q1 = self._critic1_target(next_obs, next_actions).q_value
            target_q2 = self._critic2_target(next_obs, next_actions).q_value
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        current_q1 = self._critic1(obs, actions).q_value
        current_q2 = self._critic2(obs, actions).q_value

        return F.mse_loss(current_q1, target_q), F.mse_loss(current_q2, target_q)

    def _actor_loss(self, obs: Tensor) -> Tensor:
        output = self._actor(obs)
        q_value = self._critic1(obs, output.action).q_value
        return -q_value.mean()

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._critic_loss(obs, actions, rewards, next_obs, dones)
        self._critic_optimizer.step(loss_q1 + loss_q2)

        metrics: dict[str, float] = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
        }

        self._step += 1
        if self._step % self._policy_freq == 0:
            actor_loss = self._actor_loss(obs)
            self._actor_optimizer.step(actor_loss)
            self._soft_updater.update()
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def save(self, path: Path | None = None) -> None:
        path = Path(path) if path is not None else self.default_path
        path.mkdir(parents=True, exist_ok=True)
        self._actor.save(path / 'actor.pt')
        self._critic1.save(path / 'critic1.pt')
        self._critic2.save(path / 'critic2.pt')
        self._actor_target.save(path / 'actor_target.pt')
        self._critic1_target.save(path / 'critic1_target.pt')
        self._critic2_target.save(path / 'critic2_target.pt')
        self._actor_optimizer.save(path / 'actor_optimizer.pt')
        self._critic_optimizer.save(path / 'critic_optimizer.pt')

    def load(self, path: Path) -> None:
        path = Path(path)
        self._actor.load(path / 'actor.pt')
        self._critic1.load(path / 'critic1.pt')
        self._critic2.load(path / 'critic2.pt')
        self._actor_target.load(path / 'actor_target.pt')
        self._critic1_target.load(path / 'critic1_target.pt')
        self._critic2_target.load(path / 'critic2_target.pt')
        self._actor_optimizer.load(path / 'actor_optimizer.pt')
        self._critic_optimizer.load(path / 'critic_optimizer.pt')

from typing import Any, Optional
import math

import torch


from nn_laser_stabilizer.envs.box import Box
from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.policy.policy import Policy
from nn_laser_stabilizer.config.config import Config


class RandomExplorationPolicy(Policy):
    def __init__(
        self,
        actor: Actor,
        exploration_steps: int,
    ):
        self._actor = actor
        self.exploration_steps = exploration_steps
        self.action_space = actor.action_space

        self._exploration_step_count = 0

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1
            action = self.action_space.sample()
            return action, {}
        else:
            return self._actor.act(observation, options)

    def clone(self) -> "RandomExplorationPolicy":
        cloned_actor = self._actor.clone()
        return RandomExplorationPolicy(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
        )

    def share_memory(self) -> "RandomExplorationPolicy":
        self._actor.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()

    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "RandomExplorationPolicy":
        self._actor.train(mode)
        return self

    def eval(self) -> "RandomExplorationPolicy":
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
    ) -> "RandomExplorationPolicy":
        """
        Создаёт RandomExplorationPolicy из exploration-секции конфига.

        Ожидает:
          type: \"random\"
          steps: int
        """
        steps = int(exploration_config.steps)
        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for random exploration")

        return cls(
            actor=actor,
            exploration_steps=steps,
        )


class NoisyExplorationPolicy(Policy):
    def __init__(
        self,
        actor: Actor,
        exploration_steps: int,
        policy_noise: float,
        noise_clip: float,
    ):
        self._actor = actor
        self.exploration_steps = exploration_steps
        self.action_space = actor.action_space
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self._exploration_step_count = 0

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        action, actor_options = self._actor.act(observation, options)

        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            noisy_action = self.action_space.clip(action + noise)
            return noisy_action, actor_options
        else:
            return action, actor_options

    def clone(self) -> "NoisyExplorationPolicy":
        cloned_actor = self._actor.clone()
        return NoisyExplorationPolicy(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
        )

    def share_memory(self) -> "NoisyExplorationPolicy":
        self._actor.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()

    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "NoisyExplorationPolicy":
        self._actor.train(mode)
        return self

    def eval(self) -> "NoisyExplorationPolicy":
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
    ) -> "NoisyExplorationPolicy":
        steps = int(exploration_config.steps)
        policy_noise = float(exploration_config.policy_noise)
        noise_clip = float(exploration_config.noise_clip)

        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for noisy exploration")
        if policy_noise <= 0.0:
            raise ValueError("exploration.policy_noise must be > 0 for noisy exploration")
        if noise_clip <= 0.0:
            raise ValueError("exploration.noise_clip must be > 0 for noisy exploration")

        return cls(
            actor=actor,
            exploration_steps=steps,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )


class OrnsteinUhlenbeckExplorationPolicy(Policy):
    """
    Политика исследования на основе процесса Орнштейна–Уленбека.
    Используется для добавления коррелированного во времени шума к действиям актора.

    Дискретизированный OU-процесс:
        x_{t+1} = x_t + θ(μ - x_t) * dt + σ * sqrt(dt) * N(0, 1)
    """

    def __init__(
        self,
        actor: Actor,
        exploration_steps: int,
        theta: float,
        sigma: float,
        mu: float,
        dt: float,
    ):
        self._actor = actor
        self.exploration_steps = exploration_steps
        self.action_space = actor.action_space
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt

        self._drift_coef = self.theta * self.dt
        self._diffusion_coef = self.sigma * math.sqrt(self.dt)

        self._exploration_step_count = 0
        self._ou_state: Optional[torch.Tensor] = None

    def _reset_or_get_state(self, action: torch.Tensor) -> torch.Tensor:
        if self._ou_state is None:
            self._ou_state = torch.zeros_like(action)
        return self._ou_state

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        action, actor_options = self._actor.act(observation, options)

        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1

            state = self._reset_or_get_state(action)
            noise = torch.randn_like(action)
            dx = self._drift_coef * (self.mu - state) + self._diffusion_coef * noise
            state = state + dx
            self._ou_state = state

            noisy_action = self.action_space.clip(action + state)
            return noisy_action, actor_options
        else:
            return action, actor_options

    def clone(self) -> "OrnsteinUhlenbeckExplorationPolicy":
        cloned_actor = self._actor.clone()
        # OU-состояние при клонировании не копируем, оно будет инициализировано заново
        return OrnsteinUhlenbeckExplorationPolicy(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
            theta=self.theta,
            sigma=self.sigma,
            mu=self.mu,
            dt=self.dt,
        )

    def share_memory(self) -> "OrnsteinUhlenbeckExplorationPolicy":
        self._actor.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()

    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "OrnsteinUhlenbeckExplorationPolicy":
        self._actor.train(mode)
        return self

    def eval(self) -> "OrnsteinUhlenbeckExplorationPolicy":
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
    ) -> "OrnsteinUhlenbeckExplorationPolicy":
        """
        Создаёт OrnsteinUhlenbeckExplorationPolicy из exploration-секции конфига.

        Ожидает:
          type: \"ou\"
          steps, sigma, theta, mu, dt
        """
        steps = int(exploration_config.steps)
        sigma = float(exploration_config.sigma)
        theta = float(exploration_config.theta)
        mu = float(exploration_config.mu)
        dt = float(exploration_config.dt)

        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for OU exploration")
        if sigma <= 0.0:
            raise ValueError("exploration.sigma must be > 0 for OU exploration")
        if theta <= 0.0:
            raise ValueError("exploration.theta must be > 0 for OU exploration")
        if dt <= 0.0:
            raise ValueError("exploration.dt must be > 0 for OU exploration")

        return cls(
            actor=actor,
            exploration_steps=steps,
            theta=theta,
            sigma=sigma,
            mu=mu,
            dt=dt,
        )
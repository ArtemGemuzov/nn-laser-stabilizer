from typing import Any, Optional
from abc import ABC, abstractmethod
import math

import torch

from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import ExplorationType


class Policy(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        ...
    
    @abstractmethod
    def clone(self) -> "Policy":
        ...
    
    @abstractmethod
    def share_memory(self) -> "Policy":
        ...
    
    @abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        ...
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        ...
    
    @abstractmethod
    def eval(self) -> "Policy":
        ...
    
    @abstractmethod
    def train(self, mode: bool = True) -> "Policy":
        ...
    
    @abstractmethod
    def warmup(self, observation_space: Box, num_steps: int = 100) -> None:
        """Run light inference-only warmup without affecting exploration counters."""
        ...


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


def make_policy_from_config(
    actor: Actor,
    exploration_config: Config,
) -> Policy:
    exploration_type = ExplorationType.from_str(exploration_config.type)
    exploration_steps = exploration_config.steps
    
    if exploration_type == ExplorationType.NONE:
        if exploration_steps != 0:
            raise ValueError(
                f"exploration_steps must be 0 when exploration_type is {ExplorationType.NONE}, "
                f"got exploration_steps={exploration_steps}"
            )
        return DeterministicPolicy(actor=actor)
    elif exploration_type == ExplorationType.RANDOM:
        return RandomExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
        )
    elif exploration_type == ExplorationType.NOISY:
        policy_noise = exploration_config.policy_noise
        noise_clip = exploration_config.noise_clip
        
        if policy_noise <= 0.0:
            raise ValueError(
                f"policy_noise must be greater than 0 when exploration_type is {ExplorationType.NOISY}, "
                f"got policy_noise={policy_noise}"
            )
        if noise_clip <= 0.0:
            raise ValueError(
                f"noise_clip must be greater than 0 when exploration_type is {ExplorationType.NOISY}, "
                f"got noise_clip={noise_clip}"
            )
        
        return NoisyExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )
    elif exploration_type == ExplorationType.OU:
        sigma = exploration_config.sigma
        theta = exploration_config.theta
        mu = exploration_config.mu
        dt = exploration_config.dt

        if sigma <= 0.0:
            raise ValueError(
                f"sigma must be greater than 0 when exploration_type is {ExplorationType.OU}, "
                f"got sigma={sigma}"
            )

        if theta <= 0.0:
            raise ValueError(
                f"theta must be greater than 0 when exploration_type is {ExplorationType.OU}, "
                f"got theta={theta}"
            )

        if dt <= 0.0:
            raise ValueError(
                f"dt must be greater than 0 when exploration_type is {ExplorationType.OU}, "
                f"got dt={dt}"
            )

        return OrnsteinUhlenbeckExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            theta=theta,
            sigma=sigma,
            mu=mu,
            dt=dt,
        )
    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")
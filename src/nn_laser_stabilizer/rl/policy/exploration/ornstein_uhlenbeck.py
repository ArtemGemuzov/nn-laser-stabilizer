from typing import Optional
import math

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploaration import BaseExplorationPolicy


class OrnsteinUhlenbeckExplorationPolicy(BaseExplorationPolicy):
    """
    Политика исследования на основе процесса Орнштейна–Уленбека.
    Используется для добавления коррелированного во времени шума к действиям актора.

    Дискретизированный OU-процесс:
        x_{t+1} = x_t + θ(μ - x_t) * dt + σ * sqrt(dt) * N(0, 1)
    """

    def __init__(
        self,
        inner: Policy,
        action_space: Box,
        exploration_steps: int,
        theta: float,
        sigma: float,
        mu: float,
        dt: float,
    ):
        super().__init__(inner, action_space, exploration_steps)
        self._theta = theta
        self._sigma = sigma
        self._mu = mu
        self._dt = dt
        self._drift_coef = theta * dt
        self._diffusion_coef = sigma * math.sqrt(dt)
        self._ou_state: Optional[torch.Tensor] = None

    def _get_ou_state(self, action: torch.Tensor) -> torch.Tensor:
        if self._ou_state is None:
            self._ou_state = torch.zeros_like(action)
        return self._ou_state

    def _explore(self, action: torch.Tensor, options: dict) -> torch.Tensor:
        ou_state = self._get_ou_state(action)
        noise = torch.randn_like(action)
        dx = self._drift_coef * (self._mu - ou_state) + self._diffusion_coef * noise
        ou_state = ou_state + dx
        self._ou_state = ou_state
        return self._action_space.clip(action + ou_state)

    def clone(self) -> "OrnsteinUhlenbeckExplorationPolicy":
        return OrnsteinUhlenbeckExplorationPolicy(
            inner=self._inner.clone(),
            action_space=self._action_space,
            exploration_steps=self._exploration_steps,
            theta=self._theta,
            sigma=self._sigma,
            mu=self._mu,
            dt=self._dt,
        )

    @classmethod
    def from_config(
        cls, exploration_config: Config, *, policy: Policy, action_space: Box,
    ) -> "OrnsteinUhlenbeckExplorationPolicy":
        steps = int(exploration_config.steps)
        sigma = float(exploration_config.sigma)
        theta = float(exploration_config.theta)
        mu = float(exploration_config.mu)
        dt = float(exploration_config.dt)

        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for OU exploration")
        if sigma <= 0.0:
            raise ValueError("exploration.sigma must be > 0")
        if theta <= 0.0:
            raise ValueError("exploration.theta must be > 0")
        if dt <= 0.0:
            raise ValueError("exploration.dt must be > 0")

        return cls(
            inner=policy,
            action_space=action_space,
            exploration_steps=steps,
            theta=theta,
            sigma=sigma,
            mu=mu,
            dt=dt,
        )

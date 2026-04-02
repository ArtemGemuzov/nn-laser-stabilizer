from typing import Optional
import math

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.spaces.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy
from nn_laser_stabilizer.rl.policy.exploration.base_exploration import BaseExplorationPolicy


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
        *,
        start_step: int,
        end_step: int | None,
        theta: float,
        sigma: float,
        mu: float,
        dt: float,
    ):
        super().__init__(inner, action_space, start_step=start_step, end_step=end_step)
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
            start_step=self._start_step,
            end_step=self._end_step,
            theta=self._theta,
            sigma=self._sigma,
            mu=self._mu,
            dt=self._dt,
        )

    @classmethod
    def from_config(
        cls, exploration_config: Config, *, policy: Policy, action_space: Box,
    ) -> "OrnsteinUhlenbeckExplorationPolicy":
        start_step = int(exploration_config.get("start_step", 0))
        steps = int(exploration_config.get("steps", 0))
        end_step_raw = exploration_config.get("end_step", None)
        end_step = None if end_step_raw is None else int(end_step_raw)
        sigma = float(exploration_config.sigma)
        theta = float(exploration_config.theta)
        mu = float(exploration_config.mu)
        dt = float(exploration_config.dt)

        if start_step < 0:
            raise ValueError("exploration.start_step must be >= 0 for OU exploration")
        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for OU exploration")
        if end_step is None:
            end_step = start_step + steps
        if end_step < start_step:
            raise ValueError("exploration.end_step must be >= exploration.start_step for OU exploration")
        if sigma <= 0.0:
            raise ValueError("exploration.sigma must be > 0")
        if theta <= 0.0:
            raise ValueError("exploration.theta must be > 0")
        if dt <= 0.0:
            raise ValueError("exploration.dt must be > 0")

        return cls(
            inner=policy,
            action_space=action_space,
            start_step=start_step,
            end_step=end_step,
            theta=theta,
            sigma=sigma,
            mu=mu,
            dt=dt,
        )

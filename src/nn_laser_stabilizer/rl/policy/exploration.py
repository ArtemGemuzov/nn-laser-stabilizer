from typing import Any, Optional
import math

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.utils.pid import PIDDelta
from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.model.actor import Actor
from nn_laser_stabilizer.rl.policy.policy import Policy



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


class PIDExplorationPolicy(Policy):
    CUR_ERROR_KEY = "env.cur_error"
    PREV_ERROR_KEY = "env.prev_error"
    PREV_PREV_ERROR_KEY = "env.prev_prev_error"

    def __init__(
        self,
        actor: Actor,
        exploration_steps: int,
        pid: PIDDelta,
        max_delta: float,
    ):
        self._actor = actor
        self.exploration_steps = exploration_steps
        self.action_space = actor.action_space

        self._pid = pid
        self._max_delta = max_delta

        self._exploration_step_count = 0

    def _compute_pid_action(
        self, cur_error: float, prev_error: float, prev_prev_error: float,
    ) -> torch.Tensor:
        delta = self._pid.compute_from_errors(cur_error, prev_error, prev_prev_error)
        action_value = torch.tensor([delta / self._max_delta], dtype=torch.float32)
        return torch.clamp(action_value, -1.0, 1.0)

    def act(self, observation: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._exploration_step_count < self.exploration_steps:
            self._exploration_step_count += 1

            cur_error = float(options[self.CUR_ERROR_KEY])
            prev_error = float(options[self.PREV_ERROR_KEY])
            prev_prev_error = float(options[self.PREV_PREV_ERROR_KEY])

            action = self._compute_pid_action(cur_error, prev_error, prev_prev_error)
            return action, options
        else:
            return self._actor.act(observation, options)

    def clone(self) -> "PIDExplorationPolicy":
        cloned_actor = self._actor.clone()
        pid = PIDDelta(
            kp=self._pid.kp,
            ki=self._pid.ki,
            kd=self._pid.kd,
            dt=self._pid.dt,
        )
        return PIDExplorationPolicy(
            actor=cloned_actor,
            exploration_steps=self.exploration_steps,
            pid=pid,
            max_delta=self._max_delta,
        )

    def share_memory(self) -> "PIDExplorationPolicy":
        self._actor.share_memory()
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._actor.state_dict()

    def load_state_dict(self, state_dict):
        return self._actor.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> "PIDExplorationPolicy":
        self._actor.train(mode)
        return self

    def eval(self) -> "PIDExplorationPolicy":
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
    ) -> "PIDExplorationPolicy":
        steps = int(exploration_config.steps)
        kp = float(exploration_config.kp)
        ki = float(exploration_config.ki)
        kd = float(exploration_config.kd)
        dt = float(exploration_config.dt)
        max_delta = float(exploration_config.max_delta)

        if steps < 0:
            raise ValueError("exploration.steps must be >= 0 for PID exploration")
        if dt <= 0.0:
            raise ValueError("exploration.dt must be > 0 for PID exploration")
        if max_delta <= 0.0:
            raise ValueError("exploration.max_delta must be > 0 for PID exploration")

        pid = PIDDelta(kp=kp, ki=ki, kd=kd, dt=dt)

        return cls(
            actor=actor,
            exploration_steps=steps,
            pid=pid,
            max_delta=max_delta,
        )
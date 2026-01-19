from typing import Optional

import numpy as np
import gymnasium as gym


class NNPid(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        setpoint: float = 0.0,
        dt: float = 0.1,
        k: float = 1.0,
        u_gain: float = 1.0,
        max_steps: int = 1000,
        action_low: float = -5.0,
        action_high: float = 5.0,
        process_noise_std: float = 0.0,
        integral_clip: float = 10.0,
    ):
        super().__init__()
        self.setpoint = float(setpoint)
        self.dt = float(dt)
        self.k = float(k)
        self.u_gain = float(u_gain)
        self.max_steps = int(max_steps)
        self.process_noise_std = float(process_noise_std)
        self.integral_clip = float(integral_clip)

        self.action_space = gym.spaces.Box(
            low=np.array([action_low], dtype=np.float32),
            high=np.array([action_high], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -integral_clip], dtype=np.float32),
            high=np.array([np.inf, np.inf, integral_clip], dtype=np.float32),
            dtype=np.float32,
        )

        self._x: float = 0.0
        self._error: float = 0.0
        self._prev_error: float = 0.0
        self._integral_error: float = 0.0
        self._step = 0

    def _compute_error(self) -> None:
        self._prev_error = self._error
        self._error = self.setpoint - self._x

    def _get_observation(self) -> np.ndarray:
        d_error_dt = (self._error - self._prev_error) / self.dt
        integral_clipped = np.clip(self._integral_error, -self.integral_clip, self.integral_clip)
        return np.array([self._error, d_error_dt, integral_clipped], dtype=np.float32)

    def _compute_reward(self, obs: np.ndarray) -> float:
        error, d_error_dt, integral_error = obs
        return float(-abs(error))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = float(action[0])

        noise = np.random.normal(0.0, self.process_noise_std) if self.process_noise_std > 0 else 0.0
        self._x = self._x + self.dt * (-self.k * self._x + self.u_gain * u + noise)

        self._compute_error()
        self._integral_error += self._error * self.dt
        obs = self._get_observation()
        reward = self._compute_reward(obs)

        self._step += 1
        terminated = False
        truncated = self._step >= self.max_steps
        info = {"state": self._x, "action": u}
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step = 0
        self._x = 0.0
        self._error = 0.0
        self._prev_error = 0.0
        self._integral_error = 0.0
        self._compute_error()
        obs = self._get_observation()
        return obs, {}

    def render(self):
        return {
            "state": self._x,
            "error": self._error,
            "integral_error": self._integral_error,
        }

    def close(self):
        pass
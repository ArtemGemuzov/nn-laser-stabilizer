from typing import cast

import gymnasium as gym


class RewardEMAWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, alpha: float = 0.3):
        super().__init__(env)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._ema: float | None = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = cast(float, reward)
        if self._ema is None:
            self._ema = reward
        else:
            self._ema = self._alpha * self._ema + (1.0 - self._alpha) * reward
        info["reward_ema"] = self._ema
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ema = None
        return obs, info

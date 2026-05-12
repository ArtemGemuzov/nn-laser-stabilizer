import numpy as np
import gymnasium as gym


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, max_delta: int):
        super().__init__(env)
        if max_delta <= 0:
            raise ValueError(f"max_delta must be > 0, got {max_delta}")
        self._max_delta = max_delta
        self.action_space = gym.spaces.Discrete(2 * max_delta + 1)

    def action(self, action: int) -> np.ndarray:
        delta = int(action) - self._max_delta
        norm = delta / self._max_delta
        return np.array([norm], dtype=np.float32)

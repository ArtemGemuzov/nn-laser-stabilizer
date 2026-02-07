from typing import Optional, cast

import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.base_env import BaseEnv


class PendulumNoVelEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        
        original_env = gym.make("Pendulum-v1")
        self.env = original_env
    
        self.action_space = self.env.action_space
    
        original_observation_space = original_env.observation_space
        original_observation_space_box : gym.spaces.Box = cast(gym.spaces.Box, original_observation_space)
        low = np.delete(original_observation_space_box.low, 2)
        high = np.delete(original_observation_space_box.high, 2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        filtered_obs = observation[:2]
        return filtered_obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = self.env.reset(seed=seed, options=options)
        filtered_obs = observation[:2]
        return filtered_obs, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @classmethod
    def from_config(cls, config: Config) -> "PendulumNoVelEnv":
        return cls()
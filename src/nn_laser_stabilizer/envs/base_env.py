from abc import ABC, abstractmethod

import gymnasium as gym

from nn_laser_stabilizer.config.config import Config


class BaseEnv(gym.Env, ABC):
    metadata = {"render_modes": []}

    @classmethod
    @abstractmethod
    def from_config(cls: type["BaseEnv"], config: Config) -> "BaseEnv":
        ...

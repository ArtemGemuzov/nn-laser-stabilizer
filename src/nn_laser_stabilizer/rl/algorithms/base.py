from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.policy.policy import Policy


class Agent(ABC):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "Agent": ...

    @abstractmethod
    def exploration_policy(self, exploration_config: Config) -> Policy: ...

    @abstractmethod
    def default_policy(self) -> Policy: ...

    @abstractmethod
    def save_models(self, models_dir: Path) -> None: ...


class Learner(ABC):
    @abstractmethod
    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]: ...

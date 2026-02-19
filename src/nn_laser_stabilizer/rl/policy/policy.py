from typing import Any
from abc import ABC, abstractmethod

import torch

from nn_laser_stabilizer.rl.envs.spaces.box import Box


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
        ...

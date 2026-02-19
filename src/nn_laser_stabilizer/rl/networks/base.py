from abc import abstractmethod
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def clone(self, reinitialize_weights: bool = False) -> "BaseModel":
        ...

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        path = Path(path)
        sd = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(sd)


HiddenState = tuple[Tensor, Tensor]


@dataclass
class NetworkOutput:
    output: Tensor
    state: HiddenState | None = None


class ActorNetwork(BaseModel):
    @abstractmethod
    def forward(self, observation: Tensor, state: HiddenState | None = None) -> NetworkOutput:
        ...

    @abstractmethod
    def clone(self, reinitialize_weights: bool = False) -> "ActorNetwork":
        ...


class CriticNetwork(BaseModel):
    @abstractmethod
    def forward(self, observation: Tensor, action: Tensor, state: HiddenState | None = None) -> NetworkOutput:
        ...

    @abstractmethod
    def clone(self, reinitialize_weights: bool = False) -> "CriticNetwork":
        ...

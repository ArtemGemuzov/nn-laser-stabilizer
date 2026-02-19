from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class Optimizer:  
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, **kwargs):
        self._optimizer = optim.Adam(params, lr=lr, **kwargs)
    
    def step(self, loss: torch.Tensor, set_to_none: bool = True) -> torch.Tensor:
        self._optimizer.zero_grad(set_to_none=set_to_none)
        loss.backward()
        self._optimizer.step()
        return loss.detach()

    def save(self, path: Path) -> None:
        torch.save(self._optimizer.state_dict(), path)

    def load(self, path: Path) -> None:
        self._optimizer.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    def __getattr__(self, name):
        return getattr(self._optimizer, name)


class SoftUpdater:
    """
    Класс для мягкого обновления параметров таргет-сетей:
    target_param ← lerp(target_param, source_param, tau)

    Принимает пары (target_param, source_param), чтобы не зависеть от структуры модулей.
    """

    def __init__(
        self,
        pairs: Iterable[Tuple[nn.Parameter, nn.Parameter]],
        tau: float = 0.005,
    ):
        self.tau = tau
        self._pairs: List[Tuple[nn.Parameter, nn.Parameter]] = list(pairs)
    
    def update(self) -> None:
        for target_param, source_param in self._pairs:
            target_param.data.lerp_(source_param.data, self.tau)

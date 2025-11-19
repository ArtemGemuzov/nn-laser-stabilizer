import os
import random
import copy

from typing import Tuple, List

import numpy as np

import torch
import torch.nn as nn


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(42)


def get_hydra_runtime_output_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.getcwd()


def make_target(network: nn.Module) -> nn.Module:
    target = copy.deepcopy(network)
    target.load_state_dict(network.state_dict())
    for param in target.parameters():
        param.requires_grad = False
    return target


class SoftUpdater:  
    def __init__(self, loss_module, tau: float = 0.005):
        # TODO: можно будет обобщить
        self.tau = tau
        self._pairs: List[Tuple[nn.Module, nn.Module]] = []
        
        self._register(loss_module.actor_target, loss_module.actor)
        self._register(loss_module.critic1_target, loss_module.critic1)
        self._register(loss_module.critic2_target, loss_module.critic2)
    
    def _register(self, target_network: nn.Module, source_network: nn.Module) -> None:
        self._pairs.append((target_network, source_network))
    
    @torch.no_grad()
    def update(self) -> None:
        for target_net, source_net in self._pairs:
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.copy_(
                    self.tau * source_param + (1.0 - self.tau) * target_param
                )


class Scaler(nn.Module):
    def __init__(self, low, high):
        super(Scaler, self).__init__()
        # TODO: нужно преобразовывать это где-то в другом месте
        if isinstance(low, (list, tuple, np.ndarray)):
            self.low = torch.tensor(low, dtype=torch.float32)
            self.high = torch.tensor(high, dtype=torch.float32)
        else:
            self.low = torch.tensor([low], dtype=torch.float32)
            self.high = torch.tensor([high], dtype=torch.float32)
        self.tanh = nn.Tanh()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tanh = self.tanh(tensor)
        return self.low + (tanh + 1) * (self.high - self.low) / 2
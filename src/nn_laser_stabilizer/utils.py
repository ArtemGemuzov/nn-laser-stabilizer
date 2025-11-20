import random
import copy

from typing import Tuple, List, Sequence

import numpy as np

import torch
import torch.nn as nn


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    
    def update(self) -> None:
        for target_net, source_net in self._pairs:
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.lerp_(source_param.data, self.tau)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
) -> nn.Sequential:
    layers = []
    prev_size = input_dim
    
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        prev_size = hidden_size
    
    layers.append(nn.Linear(prev_size, output_dim))
    return nn.Sequential(*layers)


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
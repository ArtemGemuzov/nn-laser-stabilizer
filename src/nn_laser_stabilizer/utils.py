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
    def __init__(self, tau: float = 0.005):
        self.tau = tau
        self._pairs: List[Tuple[nn.Module, nn.Module]] = []
    
    def register(self, target_network: nn.Module, source_network: nn.Module) -> None:
        self._pairs.append((target_network, source_network))
    
    def update(self) -> None:
        for target_net, source_net in self._pairs:
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.copy_(
                    self.tau * source_param.data + (1.0 - self.tau) * target_param.data
                )
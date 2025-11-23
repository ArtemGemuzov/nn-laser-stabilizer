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
    
    def __getattr__(self, name):
        return getattr(self._optimizer, name)


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


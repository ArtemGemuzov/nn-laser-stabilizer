from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict

import torch

from nn_laser_stabilizer.model import Model


class Policy(Model, ABC):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass
    
    @torch.no_grad()
    def act(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self(observation, options)
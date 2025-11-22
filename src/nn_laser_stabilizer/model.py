from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class Model(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_kwargs: Dict[str, Any] = kwargs
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
    
    def clone(self, reinitialize_weights: bool = False) -> "Model":
        new_model = self.__class__(**self._init_kwargs)
        if not reinitialize_weights:
            new_model.load_state_dict(self.state_dict())
        
        return new_model



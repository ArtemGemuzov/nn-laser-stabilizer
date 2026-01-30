from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_kwargs: Dict[str, Any] = kwargs
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass
    
    def clone(self, reinitialize_weights: bool = False) -> "Model":
        new_model = self.__class__(**self._init_kwargs)
        if not reinitialize_weights:
            new_model.load_state_dict(self.state_dict())
        
        return new_model
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'state_dict': self.state_dict(),
            '_init_kwargs': self._init_kwargs,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: Path) -> "Model":
        import sys
        import nn_laser_stabilizer.envs.box as new_box_mod

        sys.modules['nn_laser_stabilizer.box'] = new_box_mod

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format in {path}. "
                "Expected dict with 'state_dict' and '_init_kwargs' keys."
            )
        
        if '_init_kwargs' not in checkpoint:
            raise ValueError(
                f"Checkpoint {path} does not contain '_init_kwargs'. "
                "Cannot load model without initialization arguments."
            )
        
        state_dict = checkpoint['state_dict']
        saved_kwargs = checkpoint['_init_kwargs']
        
        model = cls(**saved_kwargs)
        model.load_state_dict(state_dict)
        return model



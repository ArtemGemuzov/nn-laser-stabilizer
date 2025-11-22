from abc import ABC, abstractmethod

import torch

from nn_laser_stabilizer.model import Model


class Policy(Model, ABC):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        pass
    
    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return self(observation)
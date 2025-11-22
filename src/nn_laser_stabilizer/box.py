import torch

import numpy as np
import gymnasium as gym


class Box:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        dim: int
    ):
        self.low : torch.Tensor = torch.from_numpy(low)
        self.high : torch.Tensor =  torch.from_numpy(high)
        
        self.dim : int = dim
        
        if self.low.shape != self.high.shape:
            raise ValueError(f"Low and high must have the same shape, got {self.low.shape} and {self.high.shape}")
        
        if not (self.low <= self.high).all():
            raise ValueError("Low must be less than or equal to high")
    
    @classmethod
    def from_gymnasium(cls, gym_box: gym.spaces.Box) -> "Box":
        if len(gym_box.shape) == 1:
            dim = gym_box.shape[0]
        else:
            dim = int(np.prod(gym_box.shape))
        
        return cls(
            low=gym_box.low,
            high=gym_box.high,
            dim=dim
        )
    
    def sample(self) -> torch.Tensor:
        uniform = torch.rand(self.dim)
        sample = self.low + uniform * (self.high - self.low)
        return sample
    
    def contains(self, x: torch.Tensor) -> bool: 
        if x.shape[-1] != self.dim:
            return False
        
        return (x >= self.low).all() and (x <= self.high).all()
    
    def clip(self, x: torch.Tensor) -> torch.Tensor:  
        return torch.clamp(x, self.low, self.high)

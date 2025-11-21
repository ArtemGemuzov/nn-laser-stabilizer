from typing import Optional, Tuple, Union

import torch
import numpy as np
import gymnasium as gym


class TorchEnvWrapper(gym.Wrapper): 
    def __init__(
        self,
        env: gym.Env,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(env)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=self.device, dtype=self.dtype)

        if isinstance(x, bool):
            return torch.tensor(x, device=self.device, dtype=torch.bool)

        if isinstance(x, (int, float, np.number)):
            return torch.tensor(x, device=self.device, dtype=self.dtype)

        raise TypeError(f"Unsupported type for _to_tensor: {type(x)}")
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = self._to_numpy(action)
        
        observation, reward, terminated, truncated, info = self.env.step(action_np)

        observation = self._to_tensor(observation)
        reward = self._to_tensor(reward)
        terminated = self._to_tensor(terminated)
        truncated = self._to_tensor(truncated)
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        observation, info = self.env.reset(seed=seed, options=options)
        observation = self._to_tensor(observation)
        return observation, info
from typing import Optional, Tuple

import torch
import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.space import Box


class TorchEnvWrapper(gym.Wrapper): 
    def __init__(
        self,
        env: gym.Env
    ):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")
        
        self.observation_space = Box.from_gymnasium(env.observation_space)
        
        if not isinstance(env.action_space, gym.spaces.Box):
             raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
        
        self.action_space = Box.from_gymnasium(env.action_space)
        
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=torch.float32)

        if isinstance(x, bool):
            return torch.tensor(x, dtype=torch.bool)

        if isinstance(x, (int, float, np.number)):
            return torch.tensor(x, dtype=torch.float32)

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
    

class PendulumNoVelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        
        self.env = gym.make("Pendulum-v1")
    
        self.action_space = self.env.action_space
        
        low = np.delete(self.env.observation_space.low, 2)
        high = np.delete(self.env.observation_space.high, 2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        filtered_obs = observation[:2]
        return filtered_obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = self.env.reset(seed=seed, options=options)
        filtered_obs = observation[:2]
        return filtered_obs, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

from typing import Optional, Tuple

import torch
import numpy as np
import gymnasium as gym

from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.env import _CUSTOM_ENV_MAP


class TorchEnvWrapper: 
    def __init__(
        self,
        env: gym.Env
    ):
        self.env : gym.Env = env

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")
        self.observation_space : Box = Box.from_gymnasium(env.observation_space)
        
        if not isinstance(env.action_space, gym.spaces.Box):
             raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
        self.action_space : Box = Box.from_gymnasium(env.action_space)
        
        self._seed: Optional[int] = None
        
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
        options: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        observation, info = self.env.reset(seed=self._seed, options=options)
        observation = self._to_tensor(observation)
        return observation, info
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
    
    @classmethod
    def wrap(cls, env: gym.Env, seed: Optional[int] = None) -> "TorchEnvWrapper":
        wrapped_env = cls(env)
        if seed is not None:
            wrapped_env.set_seed(seed)
        return wrapped_env


def make_env(
    env_name: str,
    seed: Optional[int] = None,
    **env_kwargs,
) -> TorchEnvWrapper: 
    if env_name in _CUSTOM_ENV_MAP:
        env_class = _CUSTOM_ENV_MAP[env_name]
        env = env_class(**env_kwargs)
        return TorchEnvWrapper.wrap(env, seed=seed)
   
    try:
        env = gym.make(env_name, **env_kwargs)
        return TorchEnvWrapper.wrap(env, seed=seed)
    except gym.error.UnregisteredEnv:
        raise ValueError(
            f"Unknown environment: '{env_name}'. "
            f"Custom environments: {list(_CUSTOM_ENV_MAP.keys())}"
        )

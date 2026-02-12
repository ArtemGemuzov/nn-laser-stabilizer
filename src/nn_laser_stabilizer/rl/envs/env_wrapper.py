from typing import Optional

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.envs.box import Box
from nn_laser_stabilizer.rl.envs.envs import CUSTOM_ENV_MAP


class TorchEnvWrapper: 
    def __init__(
        self,
        env: gym.Env
    ):
        self._env : gym.Env = env

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

        if isinstance(x, (bool, np.bool_)):
            return torch.tensor(x, dtype=torch.bool)

        if isinstance(x, (int, float, np.number)):
            return torch.tensor(x, dtype=torch.float32)

        raise TypeError(f"Unsupported type for _to_tensor: {type(x)}")
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()
    
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = self._to_numpy(action)
        
        observation, reward, terminated, truncated, info = self._env.step(action_np)

        observation = self._to_tensor(observation)
        reward = self._to_tensor(reward)
        terminated = self._to_tensor(terminated)
        truncated = self._to_tensor(truncated)
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        options: Optional[dict] = None
    ) -> tuple[torch.Tensor, dict]:
        observation, info = self._env.reset(seed=self._seed, options=options)
        observation = self._to_tensor(observation)
        return observation, info
    
    def close(self) -> None:
        self._env.close()
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
    
    @classmethod
    def wrap(cls, env: gym.Env, seed: Optional[int] = None) -> "TorchEnvWrapper":
        wrapped_env = cls(env)
        if seed is not None:
            wrapped_env.set_seed(seed)
        return wrapped_env


def _apply_normalize_wrappers(env: gym.Env, env_config: Config) -> gym.Env:
    normalize_obs = bool(env_config.get("normalize_obs", False))
    if normalize_obs:
        env = NormalizeObservation(env)

    normalize_reward = bool(env_config.get("normalize_reward", False))
    if normalize_reward:
        env = NormalizeReward(env)

    return env


def make_env_from_config(env_config: Config, seed: Optional[int] = None) -> TorchEnvWrapper:
    env_name = env_config.name
    if env_name in CUSTOM_ENV_MAP:
        env_class = CUSTOM_ENV_MAP[env_name]
        env = env_class.from_config(env_config)
        env = _apply_normalize_wrappers(env, env_config)
        return TorchEnvWrapper.wrap(env, seed=seed)

    args = env_config.get("args")
    env_kwargs = args.to_dict() if args is not None else {}

    try:
        env = gym.make(env_name, **env_kwargs)
        env = _apply_normalize_wrappers(env, env_config)
        return TorchEnvWrapper.wrap(env, seed=seed)
    except gym.error.UnregisteredEnv:
        raise ValueError(
            f"Unknown environment: '{env_name}'. "
            f"Custom environments: {list(CUSTOM_ENV_MAP.keys())}"
        )


def get_spaces_from_config(
    env_config: Config,
    seed: Optional[int] = None,
) -> tuple[Box, Box]:
    env = make_env_from_config(env_config, seed=seed)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space
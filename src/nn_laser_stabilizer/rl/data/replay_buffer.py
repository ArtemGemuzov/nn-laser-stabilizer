from pathlib import Path
from typing import Tuple

import torch


class ReplayBuffer:
    """
    Буфер повторений для хранения опыта.
    
    Ожидается, что будет лишь один поток-читатель и один поток-писатель.
    Тензоры всегда на CPU.
    """
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int
    ):
        self.capacity = capacity
        
        self._size = torch.zeros(1, dtype=torch.int32, device='cpu')
        self._index = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        self._observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device='cpu')
        self._actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device='cpu')
        self._rewards = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')
        self._next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device='cpu')
        self._dones = torch.zeros((capacity, 1), dtype=torch.bool, device='cpu')
    
    @property
    def observations(self) -> torch.Tensor:
        return self._observations
    
    @property
    def actions(self) -> torch.Tensor:
        return self._actions
    
    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards
    
    @property
    def next_observations(self) -> torch.Tensor:
        return self._next_observations
    
    @property
    def dones(self) -> torch.Tensor:
        return self._dones
    
    def share_memory(self) -> "ReplayBuffer":
        self._size.share_memory_()
        self._index.share_memory_()
        self._observations.share_memory_()
        self._actions.share_memory_()
        self._rewards.share_memory_()
        self._next_observations.share_memory_()
        self._dones.share_memory_()
        return self
    
    @property
    def size(self) -> int:
        return int(self._size[0].item())
    
    @size.setter
    def size(self, value: int) -> None:
        self._size[0] = value
    
    @property
    def index(self) -> int:
        return int(self._index[0].item())
    
    @index.setter
    def index(self, value: int) -> None:
        self._index[0] = value
    
    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> None: 
        """
        Ожидается, что тензоры будут отсоединены от графа вычислений.
        """
        self._observations[self.index].copy_(observation)
        self._actions[self.index].copy_(action)
        self._rewards[self.index].copy_(reward)
        self._next_observations[self.index].copy_(next_observation)
        self._dones[self.index].copy_(done)
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def __len__(self) -> int:
        return self.size
    
    def get_batch(
        self, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_observations = self.next_observations[indices]
        dones = self.dones[indices]
        return observations, actions, rewards, next_observations, dones
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'capacity': self.capacity,
            'obs_dim': self._observations.shape[1],
            'action_dim': self._actions.shape[1],
            'size': self.size,
            'index': self.index,
            'observations': self._observations,
            'actions': self._actions,
            'rewards': self._rewards,
            'next_observations': self._next_observations,
            'dones': self._dones,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: Path) -> "ReplayBuffer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ReplayBuffer file not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Invalid checkpoint format in {path}. "
                "Expected dict with buffer data."
            )
        
        required_keys = ['capacity', 'obs_dim', 'action_dim', 'size', 'index',
                        'observations', 'actions', 'rewards', 'next_observations', 'dones']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(
                f"Checkpoint {path} is missing required keys: {missing_keys}"
            )
        
        buffer = cls(
            capacity=checkpoint['capacity'],
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
        )
        
        buffer.size = checkpoint['size']
        buffer.index = checkpoint['index']
        buffer._observations.copy_(checkpoint['observations'])
        buffer._actions.copy_(checkpoint['actions'])
        buffer._rewards.copy_(checkpoint['rewards'])
        buffer._next_observations.copy_(checkpoint['next_observations'])
        buffer._dones.copy_(checkpoint['dones'])
        return buffer


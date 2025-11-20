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
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...]
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        self._size = torch.zeros(1, dtype=torch.int32, device='cpu')
        self._index = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device='cpu')
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device='cpu')
    
    def share_memory(self) -> None:
        self._size.share_memory_()
        self._index.share_memory_()
        self.observations.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.next_observations.share_memory_()
        self.dones.share_memory_()
    
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
        self.observations[self.index].copy_(observation)
        self.actions[self.index].copy_(action)
        self.rewards[self.index].copy_(reward)
        self.next_observations[self.index].copy_(next_observation)
        self.dones[self.index].copy_(done)
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        current_size = self.size
        if current_size < batch_size:
            raise ValueError(
                f"Buffer size ({current_size}) is smaller than batch size ({batch_size})"
            )
        
        indices = torch.randint(0, current_size, (batch_size,), device='cpu')
        
        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_observations = self.next_observations[indices]
        dones = self.dones[indices]
        
        return observations, actions, rewards, next_observations, dones
    
    def __len__(self) -> int:
        return self.size
    
    def save(self, path: str) -> None:
        torch.save(
            {
                'observations': self.observations[:self.size],
                'actions': self.actions[:self.size],
                'rewards': self.rewards[:self.size],
                'next_observations': self.next_observations[:self.size],
                'dones': self.dones[:self.size],
                'size': self.size,
                'obs_shape': self.obs_shape,
                'action_shape': self.action_shape,
            },
            path
        )
    
    def load(self, path: str) -> None:
        data = torch.load(path, map_location='cpu')
        
        loaded_size = int(data['size'])
        if loaded_size > self.capacity:
            raise ValueError(
                f"Loaded buffer size ({loaded_size}) exceeds capacity ({self.capacity})"
            )
            
        loaded_obs_shape = tuple(data['obs_shape'])
        if loaded_obs_shape != self.obs_shape:
            raise ValueError(
                f"Observation shape mismatch: expected {self.obs_shape}, got {loaded_obs_shape}"
            )
        
        loaded_action_shape = tuple(data['action_shape'])
        if loaded_action_shape != self.action_shape:
            raise ValueError(
                f"Action shape mismatch: expected {self.action_shape}, got {loaded_action_shape}"
            )
        
        self.observations[:loaded_size] = data['observations']
        self.actions[:loaded_size] = data['actions']
        self.rewards[:loaded_size] = data['rewards']
        self.next_observations[:loaded_size] = data['next_observations']
        self.dones[:loaded_size] = data['dones']
    
        self.size = loaded_size
        self.index = loaded_size % self.capacity


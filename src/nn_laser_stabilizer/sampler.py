from typing import Tuple

import torch

from nn_laser_stabilizer.replay_buffer import ReplayBuffer


class BatchSampler:
    def __init__(self, buffer: ReplayBuffer, batch_size: int):
        self.buffer = buffer
        self.batch_size = batch_size
        self._indices = torch.zeros((batch_size,), dtype=torch.int64, device='cpu')
    
    def sample(self) -> Tuple[torch.Tensor, ...]:
        current_size = self.buffer.size
        if current_size < self.batch_size:
            raise ValueError(
                f"Buffer size ({current_size}) is smaller than batch size ({self.batch_size})"
            )
        
        torch.randint(0, current_size, (self.batch_size,), out=self._indices)
        indices = self._indices
        
        observations = self.buffer.observations[indices]
        actions = self.buffer.actions[indices]
        rewards = self.buffer.rewards[indices]
        next_observations = self.buffer.next_observations[indices]
        dones = self.buffer.dones[indices]
        
        return observations, actions, rewards, next_observations, dones


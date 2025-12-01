from typing import Tuple, Union

import torch

from nn_laser_stabilizer.config import Config
from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.types import SamplerType


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


class BatchSequenceSampler:
    def __init__(self, buffer: ReplayBuffer, batch_size: int, seq_len: int):
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len

        self._seq_indices = torch.arange(seq_len)
        self._start_indices = torch.zeros((batch_size,), dtype=torch.int64)

    def sample(self) -> Tuple[torch.Tensor, ...]:
        current_size = self.buffer.size
        if current_size < self.seq_len:
            raise ValueError(
                f"Buffer size ({current_size}) is smaller than seq_len ({self.seq_len})"
            )

        max_start = current_size - self.seq_len + 1
        if max_start < self.batch_size:
            raise ValueError(
                f"Not enough sequences in buffer. "
                f"Can create at most {max_start} sequences, but batch_size is {self.batch_size}"
            )

        torch.randint(0, max_start, (self.batch_size,), out=self._start_indices)
        indices = self._start_indices.unsqueeze(1) + self._seq_indices.unsqueeze(0)

        observations = self.buffer.observations[indices]
        actions = self.buffer.actions[indices]
        rewards = self.buffer.rewards[indices]
        next_observations = self.buffer.next_observations[indices]
        dones = self.buffer.dones[indices]
        return observations, actions, rewards, next_observations, dones
    

def make_sampler_from_config(
    sampler_config: Config,
    buffer: ReplayBuffer, 
) -> Union[BatchSampler, BatchSequenceSampler]:
    batch_size = sampler_config.batch_size
    sampler_type_str = sampler_config.type
    sampler_type = SamplerType.from_str(sampler_type_str)
    
    if sampler_type == SamplerType.SINGLE:
        return BatchSampler(buffer=buffer, batch_size=batch_size)
    elif sampler_type == SamplerType.SEQUENCE:
        seq_len = sampler_config.seq_len
        return BatchSequenceSampler(buffer=buffer, batch_size=batch_size, seq_len=seq_len)
    else:
        raise ValueError(f"Unhandled sampler type: {sampler_type}")

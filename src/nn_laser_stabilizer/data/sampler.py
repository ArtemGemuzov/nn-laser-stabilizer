from typing import Tuple, Union

import torch

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.config.types import SamplerType


class BatchSampler:
    def __init__(self, buffer: ReplayBuffer, batch_size: int):
        self.buffer = buffer
        self.batch_size = batch_size

        self._indices = torch.zeros((batch_size,), dtype=torch.int64, device='cpu')
    
    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_size = self.buffer.size
        if current_size < self.batch_size:
            raise ValueError(
                f"Buffer size ({current_size}) is smaller than batch size ({self.batch_size})"
            )
        
        torch.randint(0, current_size, (self.batch_size,), out=self._indices)
        return self.buffer.get_batch(self._indices)

    @classmethod
    def from_config(
        cls,
        sampler_config: Config,
        *,
        buffer: ReplayBuffer,
    ) -> "BatchSampler":
        """
        Создаёт BatchSampler из sampler-секции конфига.

        Ожидает:
          type: \"single\"
          batch_size: int
        """
        batch_size = int(sampler_config.batch_size)
        if batch_size <= 0:
            raise ValueError("sampler.batch_size must be > 0 for single sampler")

        return cls(buffer=buffer, batch_size=batch_size)


class BatchSequenceSampler:
    def __init__(self, buffer: ReplayBuffer, batch_size: int, seq_len: int):
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len

        self._seq_indices = torch.arange(seq_len)
        self._start_indices = torch.zeros((batch_size,), dtype=torch.int64)

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self.buffer.get_batch(indices)

    @classmethod
    def from_config(
        cls,
        sampler_config: Config,
        *,
        buffer: ReplayBuffer,
    ) -> "BatchSequenceSampler":
        """
        Создаёт BatchSequenceSampler из sampler-секции конфига.

        Ожидает:
          type: \"sequence\"
          batch_size: int
          seq_len: int
        """
        batch_size = int(sampler_config.batch_size)
        seq_len = int(sampler_config.seq_len)

        if batch_size <= 0:
            raise ValueError("sampler.batch_size must be > 0 for sequence sampler")
        if seq_len <= 0:
            raise ValueError("sampler.seq_len must be > 0 for sequence sampler")

        return cls(buffer=buffer, batch_size=batch_size, seq_len=seq_len)

def make_sampler_from_config(
    sampler_config: Config,
    buffer: ReplayBuffer, 
) -> Union[BatchSampler, BatchSequenceSampler]:
    sampler_type_str = sampler_config.type
    sampler_type = SamplerType.from_str(sampler_type_str)
    
    if sampler_type == SamplerType.SINGLE:
        return BatchSampler.from_config(sampler_config, buffer=buffer)
    elif sampler_type == SamplerType.SEQUENCE:
        return BatchSequenceSampler.from_config(sampler_config, buffer=buffer)
    else:
        raise ValueError(f"Unhandled sampler type: {sampler_type}")

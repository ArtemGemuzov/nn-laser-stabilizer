from typing import Sequence, Any, cast
from abc import ABC, abstractmethod
from pathlib import Path


import torch
import torch.nn as nn

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import NetworkType
from nn_laser_stabilizer.rl.model.model import Model
from nn_laser_stabilizer.rl.model.layers import build_mlp


class Critic(Model, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        ...

    @torch.no_grad()
    def value(self, observation: torch.Tensor, action: torch.Tensor, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if options is None:
            options = {}
        return self(observation, action, options)

    def clone(self, reinitialize_weights: bool = False) -> "Critic":
        return cast(Critic, super().clone(reinitialize_weights))

    @classmethod
    def load(cls, path: Path) -> "Critic":
        return cast(Critic, super().load(path))
    

class MLPCritic(Critic):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.net = build_mlp(
            obs_dim + action_dim,
            1,
            hidden_sizes,
        )
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if options is None:
            options = {}
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Observation and action must have matching leading dimensions, "
                f"got observation.shape={tuple(observation.shape)}, action.shape={tuple(action.shape)}"
            )
        q_value = self.net(torch.cat([observation, action], dim=-1))
        return q_value, options

    @classmethod
    def from_config(
        cls,
        network_config: Config,
        *,
        obs_dim: int,
        action_dim: int,
    ) -> "MLPCritic":
        """
        Создаёт MLPCritic из network-секции конфига.

        Ожидает:
          type: \"mlp\"
          mlp_hidden_sizes: последовательность целых
        """
        hidden_sizes_raw = network_config.mlp_hidden_sizes
        try:
            hidden_sizes_seq = tuple(int(h) for h in hidden_sizes_raw)
        except TypeError:
            raise TypeError("network.mlp_hidden_sizes must be an iterable of integers")

        if not hidden_sizes_seq:
            raise ValueError("network.mlp_hidden_sizes must be non-empty for MLPCritic")

        return cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes_seq,
        )


class LSTMCritic(Critic):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        mlp_hidden_sizes: Sequence[int] = (256,),
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_sizes=mlp_hidden_sizes,
        )
        self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.net = build_mlp(
            lstm_hidden_size + action_dim,
            1,
            mlp_hidden_sizes,
        )
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if options is None:
            options = {}
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Observation and action must have matching leading dimensions, "
                f"got observation.shape={tuple(observation.shape)}, action.shape={tuple(action.shape)}"
            )
        
        hidden_state = options.get('hidden_state')

        was_1d = observation.dim() == 1
        was_2d = observation.dim() == 2
        
        # 1D: (obs_dim) -> (1, 1, obs_dim)
        # 2D: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
        # 3D: (batch_size, seq_len, obs_dim) -> already correct
        if was_1d:
            observation = observation.unsqueeze(0).unsqueeze(1)  # (1, 1, obs_dim)
        elif observation.dim() == 2:
            observation = observation.unsqueeze(1)  # (batch_size, 1, obs_dim)

        if was_1d == 1:
            action = action.unsqueeze(0).unsqueeze(1)  # (1, 1, action_dim)
        elif was_2d == 2:
            action = action.unsqueeze(1)  # (batch_size, 1, action_dim)

        lstm_out, hidden_state = self.lstm(observation, hidden_state)  # (batch_size, seq_len, lstm_hidden_size)

        batch_size, seq_len, hidden_size = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(-1, hidden_size)  # (batch_size * seq_len, lstm_hidden_size)
        action_flat = action.reshape(-1, action.shape[-1])  # (batch_size * seq_len, action_dim)
        summary_action_flat = torch.cat([lstm_out_flat, action_flat], dim=-1)  # (batch_size * seq_len, lstm_hidden_size + action_dim)
        q_values_flat = self.net(summary_action_flat)  # (batch_size * seq_len, 1)
        q_values = q_values_flat.reshape(batch_size, seq_len, 1)  # (batch_size, seq_len, 1)
        
        if was_1d:
            q_values = q_values.squeeze(0).squeeze(0)  # scalar
        elif was_2d:
            q_values = q_values.squeeze(1)  # (batch_size, 1)
        
        options['hidden_state'] = hidden_state
        return q_values, options

    @classmethod
    def from_config(
        cls,
        network_config: Config,
        *,
        obs_dim: int,
        action_dim: int,
    ) -> "LSTMCritic":
        lstm_hidden_size = int(network_config.lstm_hidden_size)
        lstm_num_layers = int(network_config.lstm_num_layers)
        if lstm_hidden_size <= 0:
            raise ValueError("network.lstm_hidden_size must be > 0 for LSTMCritic")
        if lstm_num_layers <= 0:
            raise ValueError("network.lstm_num_layers must be > 0 for LSTMCritic")

        hidden_sizes_raw = network_config.mlp_hidden_sizes
        try:
            mlp_hidden_sizes_seq = tuple(int(h) for h in hidden_sizes_raw)
        except TypeError:
            raise TypeError("network.mlp_hidden_sizes must be an iterable of integers")

        if not mlp_hidden_sizes_seq:
            raise ValueError("network.mlp_hidden_sizes must be non-empty for LSTMCritic")

        return cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_sizes=mlp_hidden_sizes_seq,
        )

def make_critic_from_config(network_config: Config, obs_dim: int, action_dim: int) -> Critic:
    network_type = NetworkType.from_str(network_config.type)
    if network_type == NetworkType.MLP:
        return MLPCritic.from_config(
            network_config=network_config,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    elif network_type == NetworkType.LSTM:
        return LSTMCritic.from_config(
            network_config=network_config,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    else:
        raise ValueError(f"Unhandled network type: {network_type}")
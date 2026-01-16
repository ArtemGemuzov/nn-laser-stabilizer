from abc import ABC, abstractmethod

from typing import Sequence, Any, Optional

import torch
import torch.nn as nn

from nn_laser_stabilizer.model import Model
from nn_laser_stabilizer.layers import build_mlp
from nn_laser_stabilizer.experiment.config import Config
from nn_laser_stabilizer.types import NetworkType


class Critic(Model, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        ...

    @torch.no_grad()
    def value(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[dict[str, Any]] = None) -> tuple[torch.Tensor, dict[str, Any]]:
        return self(observation, action, options)
    

class MLPCritic(Critic):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256)
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.net = build_mlp(
            obs_dim + action_dim,
            1,
            hidden_sizes,
        )
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Observation and action must have matching leading dimensions, "
                f"got observation.shape={tuple(observation.shape)}, action.shape={tuple(action.shape)}"
            )
        q_value = self.net(torch.cat([observation, action], dim=-1))
        return q_value, options


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
        options: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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
    

def make_critic_from_config(network_config: Config, obs_dim: int, action_dim: int) -> Critic:
    network_type_str = network_config.type
    
    try:
        network_type = NetworkType(network_type_str)
    except ValueError:
        raise ValueError(
            f"Unknown network type: '{network_type_str}'. "
            f"Supported types: {[t.value for t in NetworkType]}"
        )
    
    if network_type == NetworkType.MLP:
        return MLPCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(network_config.mlp_hidden_sizes),
        )
    elif network_type == NetworkType.LSTM:
        return LSTMCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=network_config.lstm_hidden_size,
            lstm_num_layers=network_config.lstm_num_layers,
            mlp_hidden_sizes=tuple(network_config.mlp_hidden_sizes),
        )
    else:
        raise ValueError(f"Unhandled network type: {network_type}")
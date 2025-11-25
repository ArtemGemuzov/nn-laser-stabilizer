from abc import ABC, abstractmethod

from typing import Sequence, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from nn_laser_stabilizer.model import Model
from nn_laser_stabilizer.layers import build_mlp


class Critic(Model, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    @torch.no_grad()
    def value(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
        self.lstm = nn.LSTM(obs_dim + action_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.net = build_mlp(
            lstm_hidden_size,
            1,
            mlp_hidden_sizes,
        )
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if options is None:
            options = {}
        hidden_state = options.get('hidden_state')
        
        if observation.dim() == 2 and action.dim() == 2:
            observation = observation.unsqueeze(1)  # (batch_size, 1, obs_dim)
            action = action.unsqueeze(1)  # (batch_size, 1, action_dim)
        
        observation_action_pairs = torch.cat([observation, action], dim=-1)  # (batch_size, seq_len, obs_dim + action_dim)
        lstm_out, hidden_state = self.lstm(observation_action_pairs, hidden_state)  # (batch_size, seq_len, lstm_hidden_size)
        
        lstm_last = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        q_values = self.net(lstm_last)  # (batch_size, 1)
        
        options['hidden_state'] = hidden_state
        return q_values, options
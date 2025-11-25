from typing import Sequence, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from nn_laser_stabilizer.layers import build_mlp
from nn_laser_stabilizer.model import Model
from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.layers import Scaler
from nn_laser_stabilizer.config import Config
from nn_laser_stabilizer.types import NetworkType


class Actor(Model):
    @torch.no_grad()
    def act(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self(observation, options)


class MLPActor(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space: Box,
        hidden_sizes: Sequence[int] = (256, 256)
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, action_space=action_space, hidden_sizes=hidden_sizes)
        self.net_body = build_mlp(obs_dim, action_dim, hidden_sizes)
        self.scaler = Scaler(action_space)
    
    def forward(self, observation: torch.Tensor, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        action = self.scaler(self.net_body(observation))
        return action, options


class LSTMActor(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space: Box,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        mlp_hidden_sizes: Sequence[int] = (256,),
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_space=action_space,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_sizes=mlp_hidden_sizes,
        )
        self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.net_body = build_mlp(lstm_hidden_size, action_dim, mlp_hidden_sizes)
        self.scaler = Scaler(action_space)
    
    def forward(
        self,
        observation: torch.Tensor,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if options is None:
            options = {}
        hidden_state = options.get('hidden_state')
        
        was_1d = observation.dim() == 1
        # 1D: (obs_dim) -> (1, 1, obs_dim)
        # 2D: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
        # 3D: (batch_size, seq_len, obs_dim) -> already correct
        if observation.dim() == 1:
            observation = observation.unsqueeze(0).unsqueeze(1)  # (1, 1, obs_dim)
        elif observation.dim() == 2:
            observation = observation.unsqueeze(1)  # (batch_size, 1, obs_dim)
        # If dim() == 3, it's already in the correct format
        
        lstm_out, hidden_state = self.lstm(observation, hidden_state)  # (batch_size, seq_len, lstm_hidden_size)
        
        lstm_last = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        actions = self.net_body(lstm_last)  # (batch_size, action_dim)
        actions = self.scaler(actions)  # (batch_size, action_dim)
        
        if was_1d:
            actions = actions.squeeze(0)  # (action_dim)
        
        options['hidden_state'] = hidden_state
        return actions, options
    

def make_actor_from_config(network_config: Config, action_space: Box, observation_space: Box) -> "Actor":
    network_type_str = network_config.type
    
    try:
        network_type = NetworkType(network_type_str)
    except ValueError:
        raise ValueError(
            f"Unknown network type: '{network_type_str}'. "
            f"Supported types: {[t.value for t in NetworkType]}"
        )
    
    if network_type == NetworkType.MLP:
        return MLPActor(
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
            action_space=action_space,
            hidden_sizes=tuple(network_config.mlp_hidden_sizes),
        )
    elif network_type == NetworkType.LSTM:
        return LSTMActor(
            obs_dim=observation_space.dim,
            action_dim=action_space.dim,
            action_space=action_space,
            lstm_hidden_size=network_config.lstm_hidden_size,
            lstm_num_layers=network_config.lstm_num_layers,
            mlp_hidden_sizes=tuple(network_config.mlp_hidden_sizes),
        )
    else:
        raise ValueError(f"Unhandled network type: {network_type}")


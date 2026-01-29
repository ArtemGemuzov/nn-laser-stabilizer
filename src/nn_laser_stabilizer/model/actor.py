from typing import Sequence, Any, cast
from pathlib import Path

import torch
import torch.nn as nn

from nn_laser_stabilizer.layers import build_mlp
from nn_laser_stabilizer.model.model import Model
from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.layers import Scaler
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import NetworkType


class Actor(Model):
    def __init__(self, *, action_space: Box, **kwargs: Any):
        super().__init__(action_space=action_space, **kwargs)
        self._action_space = action_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    @torch.no_grad()
    def act(self, observation: torch.Tensor, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if options is None:
            options = {}
        return self(observation, options)
    
    def clone(self, reinitialize_weights: bool = False) -> "Actor":
        return cast(Actor, super().clone(reinitialize_weights))
    
    @classmethod
    def load(cls, path: Path) -> "Actor":
        return cast(Actor, super().load(path))


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
    
    def forward(self, observation: torch.Tensor, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if options is None:
            options = {}
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
        options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if options is None:
            options = {}
        hidden_state = options.get('hidden_state')
        
        was_1d = observation.dim() == 1
        was_2d = observation.dim() == 2
        # 1D: (obs_dim) -> (1, 1, obs_dim)
        # 2D: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
        # 3D: (batch_size, seq_len, obs_dim) -> already correct
        if was_1d:
            observation = observation.unsqueeze(0).unsqueeze(1)  # (1, 1, obs_dim)
        elif was_2d:
            observation = observation.unsqueeze(1)  # (batch_size, 1, obs_dim)
        
        lstm_out, hidden_state = self.lstm(observation, hidden_state)  # (batch_size, seq_len, lstm_hidden_size)
        
        batch_size, seq_len, hidden_size = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(-1, hidden_size)  # (batch_size * seq_len, lstm_hidden_size)
        actions_flat = self.net_body(lstm_out_flat)  # (batch_size * seq_len, action_dim)
        actions_flat = self.scaler(actions_flat)  # (batch_size * seq_len, action_dim)
        actions = actions_flat.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, action_dim)
        
        if was_1d:
            actions = actions.squeeze(0).squeeze(0)  # (action_dim)
        elif was_2d:
            actions = actions.squeeze(1)  # (batch_size, action_dim)
        
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


def load_actor_from_path(actor_path: Path, network_type: NetworkType) -> Actor:
    actor_path = Path(actor_path).resolve()
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor model not found: {actor_path}")

    if network_type == NetworkType.MLP:
        return MLPActor.load(actor_path)
    elif network_type == NetworkType.LSTM:
        return LSTMActor.load(actor_path)
    else:
        raise ValueError(f"Unhandled network type: {network_type}")

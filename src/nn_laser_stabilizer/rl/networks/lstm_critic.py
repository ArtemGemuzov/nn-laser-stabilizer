import torch
import torch.nn as nn
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.networks.layers import build_mlp
from nn_laser_stabilizer.rl.networks.base import CriticNetwork, NetworkOutput, HiddenState


class LSTMCriticNetwork(CriticNetwork):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        mlp_hidden_sizes: tuple[int, ...] = (256,),
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_num_layers = lstm_num_layers
        self._mlp_hidden_sizes = mlp_hidden_sizes
        self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.mlp = build_mlp(lstm_hidden_size + action_dim, 1, mlp_hidden_sizes)

    def forward(self, obs: Tensor, action: Tensor, state: HiddenState | None = None) -> NetworkOutput:
        if obs.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Observation and action must have matching leading dimensions, "
                f"got observation.shape={tuple(obs.shape)}, action.shape={tuple(action.shape)}"
            )

        was_1d = obs.dim() == 1
        was_2d = obs.dim() == 2

        # 1D: (obs_dim) -> (1, 1, obs_dim)
        # 2D: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
        # 3D: (batch_size, seq_len, obs_dim) -> already correct
        if was_1d:
            obs = obs.unsqueeze(0).unsqueeze(1)  # (1, 1, obs_dim)
            action = action.unsqueeze(0).unsqueeze(1)  # (1, 1, action_dim)
        elif was_2d:
            obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
            action = action.unsqueeze(1)  # (batch_size, 1, action_dim)

        lstm_out, new_state = self.lstm(obs, state)  # (batch_size, seq_len, lstm_hidden_size)

        batch_size, seq_len, hidden_size = lstm_out.shape
        lstm_flat = lstm_out.reshape(-1, hidden_size)  # (batch_size * seq_len, lstm_hidden_size)
        action_flat = action.reshape(-1, action.shape[-1])  # (batch_size * seq_len, action_dim)
        combined = torch.cat([lstm_flat, action_flat], dim=-1)  # (batch_size * seq_len, lstm_hidden_size + action_dim)
        q_flat = self.mlp(combined)  # (batch_size * seq_len, 1)
        q_values = q_flat.reshape(batch_size, seq_len, 1)  # (batch_size, seq_len, 1)

        if was_1d:
            q_values = q_values.squeeze(0).squeeze(0)  # scalar
        elif was_2d:
            q_values = q_values.squeeze(1)  # (batch_size, 1)

        return NetworkOutput(output=q_values, state=new_state)

    @classmethod
    def from_config(
        cls,
        network_config: Config,
        *,
        obs_dim: int,
        action_dim: int,
    ) -> "LSTMCriticNetwork":
        lstm_hidden_size = int(network_config.lstm_hidden_size)
        lstm_num_layers = int(network_config.lstm_num_layers)

        if lstm_hidden_size <= 0:
            raise ValueError("network.lstm_hidden_size must be > 0")
        if lstm_num_layers <= 0:
            raise ValueError("network.lstm_num_layers must be > 0")

        hidden_sizes_raw = network_config.mlp_hidden_sizes
        try:
            mlp_hidden_sizes = tuple(int(h) for h in hidden_sizes_raw)
        except TypeError:
            raise TypeError("network.mlp_hidden_sizes must be an iterable of integers")
        if not mlp_hidden_sizes:
            raise ValueError("network.mlp_hidden_sizes must be non-empty")

        return cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_hidden_sizes=mlp_hidden_sizes,
        )

    def clone(self, reinitialize_weights: bool = False) -> "LSTMCriticNetwork":
        new_net = LSTMCriticNetwork(
            self._obs_dim, self._action_dim,
            self._lstm_hidden_size, self._lstm_num_layers, self._mlp_hidden_sizes,
        )
        if not reinitialize_weights:
            new_net.load_state_dict(self.state_dict())
        return new_net

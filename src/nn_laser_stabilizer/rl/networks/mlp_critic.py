import torch
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.networks.layers import build_mlp
from nn_laser_stabilizer.rl.networks.base import CriticNetwork, NetworkOutput, HiddenState


class MLPCriticNetwork(CriticNetwork):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_sizes = hidden_sizes
        self.net = build_mlp(obs_dim + action_dim, 1, hidden_sizes)

    def forward(self, obs: Tensor, action: Tensor, state: HiddenState | None = None) -> NetworkOutput:
        if obs.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Observation and action must have matching leading dimensions, "
                f"got observation.shape={tuple(obs.shape)}, action.shape={tuple(action.shape)}"
            )
        q_value = self.net(torch.cat([obs, action], dim=-1))
        return NetworkOutput(output=q_value)

    @classmethod
    def from_config(
        cls,
        network_config: Config,
        *,
        obs_dim: int,
        action_dim: int,
    ) -> "MLPCriticNetwork":
        hidden_sizes_raw = network_config.mlp_hidden_sizes
        try:
            hidden_sizes = tuple(int(h) for h in hidden_sizes_raw)
        except TypeError:
            raise TypeError("network.mlp_hidden_sizes must be an iterable of integers")
        if not hidden_sizes:
            raise ValueError("network.mlp_hidden_sizes must be non-empty")

        return cls(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)

    def clone(self, reinitialize_weights: bool = False) -> "MLPCriticNetwork":
        new_net = MLPCriticNetwork(self._obs_dim, self._action_dim, self._hidden_sizes)
        if not reinitialize_weights:
            new_net.load_state_dict(self.state_dict())
        return new_net

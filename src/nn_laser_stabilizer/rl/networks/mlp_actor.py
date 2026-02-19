from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.networks.layers import build_mlp
from nn_laser_stabilizer.rl.networks.base import ActorNetwork, NetworkOutput, HiddenState


class MLPActorNetwork(ActorNetwork):
    def __init__(
        self,
        obs_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes
        self.net = build_mlp(obs_dim, output_dim, hidden_sizes)

    def forward(self, obs: Tensor, state: HiddenState | None = None) -> NetworkOutput:
        return NetworkOutput(output=self.net(obs))

    def clone(self, reinitialize_weights: bool = False) -> "MLPActorNetwork":
        new_net = MLPActorNetwork(self._obs_dim, self._output_dim, self._hidden_sizes)
        if not reinitialize_weights:
            new_net.load_state_dict(self.state_dict())
        return new_net

    @classmethod
    def from_config(
        cls,
        network_config: Config,
        *,
        obs_dim: int,
        output_dim: int,
    ) -> "MLPActorNetwork":
        hidden_sizes_raw = network_config.mlp_hidden_sizes
        try:
            hidden_sizes = tuple(int(h) for h in hidden_sizes_raw)
        except TypeError:
            raise TypeError("network.mlp_hidden_sizes must be an iterable of integers")
        if not hidden_sizes:
            raise ValueError("network.mlp_hidden_sizes must be non-empty")

        return cls(obs_dim=obs_dim, output_dim=output_dim, hidden_sizes=hidden_sizes)

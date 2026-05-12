from dataclasses import dataclass

from torch import Tensor

from nn_laser_stabilizer.rl.networks.base import BaseModel, ActorNetwork, HiddenState


@dataclass
class QNetworkOutput:
    q_values: Tensor
    state: HiddenState | None = None


class DiscreteQNetwork(BaseModel):
    def __init__(self, network: ActorNetwork, num_actions: int):
        super().__init__()
        self._network = network
        self._num_actions = num_actions

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def forward(self, obs: Tensor, state: HiddenState | None = None) -> QNetworkOutput:
        net_out = self._network(obs, state)
        return QNetworkOutput(q_values=net_out.output, state=net_out.state)

    def clone(self, reinitialize_weights: bool = False) -> "DiscreteQNetwork":
        cloned_network = self._network.clone(reinitialize_weights)
        new_q = DiscreteQNetwork(network=cloned_network, num_actions=self._num_actions)
        if not reinitialize_weights:
            new_q.load_state_dict(self.state_dict())
        return new_q

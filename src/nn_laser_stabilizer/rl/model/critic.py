from dataclasses import dataclass

from torch import Tensor

from nn_laser_stabilizer.rl.networks.base import BaseModel, CriticNetwork, HiddenState


@dataclass
class CriticOutput:
    q_value: Tensor
    state: HiddenState | None = None


class Critic(BaseModel):
    def __init__(self, network: CriticNetwork):
        super().__init__()
        self._network = network

    def forward(self, obs: Tensor, action: Tensor, state: HiddenState | None = None) -> CriticOutput:
        net_out = self._network(obs, action, state)
        return CriticOutput(q_value=net_out.output, state=net_out.state)

    def clone(self, reinitialize_weights: bool = False) -> "Critic":
        cloned_network = self._network.clone(reinitialize_weights)
        new_critic = Critic(network=cloned_network)
        if not reinitialize_weights:
            new_critic.load_state_dict(self.state_dict())
        return new_critic

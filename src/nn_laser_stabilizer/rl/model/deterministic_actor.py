from dataclasses import dataclass

from torch import Tensor

from nn_laser_stabilizer.rl.networks.base import BaseModel, ActorNetwork, HiddenState
from nn_laser_stabilizer.rl.networks.scaler import Scaler
from nn_laser_stabilizer.rl.envs.spaces.box import Box


@dataclass
class DeterministicActorOutput:
    action: Tensor
    raw_action: Tensor
    state: HiddenState | None = None


class DeterministicActor(BaseModel):
    def __init__(self, network: ActorNetwork, action_space: Box):
        super().__init__()
        self._network = network
        self._scaler = Scaler(action_space)
        self._action_space = action_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    def forward(self, obs: Tensor, state: HiddenState | None = None) -> DeterministicActorOutput:
        net_out = self._network(obs, state)
        action = self._scaler(net_out.output)
        return DeterministicActorOutput(
            action=action,
            raw_action=net_out.output,
            state=net_out.state,
        )

    def clone(self, reinitialize_weights: bool = False) -> "DeterministicActor":
        cloned_network = self._network.clone(reinitialize_weights)
        new_actor = DeterministicActor(network=cloned_network, action_space=self._action_space)
        if not reinitialize_weights:
            new_actor.load_state_dict(self.state_dict())
        return new_actor

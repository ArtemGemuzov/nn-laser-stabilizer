from dataclasses import dataclass

import torch
from torch import Tensor

from nn_laser_stabilizer.rl.networks.base import BaseModel, ActorNetwork, HiddenState, NetworkOutput
from nn_laser_stabilizer.rl.networks.scaler import Scaler
from nn_laser_stabilizer.rl.envs.spaces.box import Box

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class StochasticActorOutput:
    action: Tensor
    mean_action: Tensor
    log_prob: Tensor
    raw_action: Tensor
    state: HiddenState | None = None


def gaussian_log_prob(
    normal: torch.distributions.Normal,
    raw_action: Tensor,
    action_scale: Tensor,
) -> Tensor:
    log_prob = normal.log_prob(raw_action)
    log_prob = log_prob - torch.log(action_scale * (1.0 - torch.tanh(raw_action).pow(2)) + 1e-6)
    return log_prob.sum(dim=-1, keepdim=True)


class StochasticActor(BaseModel):
    def __init__(self, network: ActorNetwork, action_space: Box):
        super().__init__()
        self._network = network
        self._scaler = Scaler(action_space)
        self._action_space = action_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    def forward(self, obs: Tensor, state: HiddenState | None = None) -> StochasticActorOutput:
        net_out: NetworkOutput = self._network(obs, state)
        mean, log_std = net_out.output.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = self._scaler(x_t)
        mean_action = self._scaler(mean)
        action_scale = (self._action_space.high - self._action_space.low) / 2.0
        log_prob = gaussian_log_prob(normal, x_t, action_scale)

        return StochasticActorOutput(
            action=action,
            mean_action=mean_action,
            log_prob=log_prob,
            raw_action=x_t,
            state=net_out.state,
        )

    def clone(self, reinitialize_weights: bool = False) -> "StochasticActor":
        cloned_network = self._network.clone(reinitialize_weights)
        new_actor = StochasticActor(network=cloned_network, action_space=self._action_space)
        if not reinitialize_weights:
            new_actor.load_state_dict(self.state_dict())
        return new_actor

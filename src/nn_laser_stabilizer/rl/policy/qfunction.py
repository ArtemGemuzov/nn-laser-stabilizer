import torch
from torch import Tensor

from nn_laser_stabilizer.rl.model.critic import Critic


class QFunction:
    def __init__(self, critic: Critic):
        self._critic = critic

    @property
    def critic(self) -> Critic:
        return self._critic

    @torch.no_grad()
    def evaluate(self, obs: Tensor, action: Tensor) -> Tensor:
        return self._critic(obs, action).q_value

import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.bc.agent import BCAgent


class BCLoss:
    def __init__(self, agent: BCAgent):
        self._agent = agent

    @classmethod
    def from_config(cls, algorithm_config: Config, agent: BCAgent) -> "BCLoss":
        return cls(agent=agent)

    def loss(self, obs: Tensor, actions: Tensor) -> Tensor:
        output = self._agent.actor(obs)
        return F.mse_loss(output.action, actions)

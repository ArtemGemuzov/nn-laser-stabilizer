from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config


class TD3Loss:
    def __init__(self, gamma: float):
        self._gamma = gamma

    @classmethod
    def from_config(cls, algorithm_config: Config) -> "TD3Loss":
        gamma = float(algorithm_config.gamma)
        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")
        return cls(gamma=gamma)

    def critic_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        current_q1 = output["current_q1"]
        current_q2 = output["current_q2"]
        target_q1 = output["target_q1"]
        target_q2 = output["target_q2"]
        rewards = output["rewards"]
        dones = output["dones"]

        with torch.no_grad():
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        return {
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
        }

    def actor_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        actor_q_value = output["actor_q_value"]
        actor_loss = -actor_q_value.mean()

        return {
            "actor_loss": actor_loss,
        }

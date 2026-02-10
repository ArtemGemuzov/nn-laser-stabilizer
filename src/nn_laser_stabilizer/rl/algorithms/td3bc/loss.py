from typing import Any

import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.td3.loss import TD3Loss


class TD3BCLoss(TD3Loss):
    def __init__(self, gamma: float, alpha: float):
        super().__init__(gamma=gamma)
        self._alpha = alpha

    @classmethod
    def from_config(cls, algorithm_config: Config) -> "TD3BCLoss":
        gamma = float(algorithm_config.gamma)
        alpha = float(algorithm_config.alpha)
        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")
        return cls(gamma=gamma, alpha=alpha)

    def actor_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        actor_q_value = output["actor_q_value"]
        actor_actions = output["actor_actions"]
        dataset_actions = output["dataset_actions"]
        lambda_coef = output["lambda_coef"]

        td3_term = -lambda_coef * actor_q_value.mean()
        bc_term = self._alpha * F.mse_loss(actor_actions, dataset_actions)
        actor_loss = td3_term + bc_term

        return {
            "actor_loss": actor_loss,
            "actor_td3_term": td3_term,
            "actor_bc_term": bc_term,
        }

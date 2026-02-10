from typing import Any

import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config


class BCLoss:
    @classmethod
    def from_config(cls, algorithm_config: Config) -> "BCLoss":
        return cls()

    def actor_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        actions = output["actions"]
        dataset_actions = output["dataset_actions"]
        loss = F.mse_loss(actions, dataset_actions)

        return {
            "actor_loss": loss,
        }

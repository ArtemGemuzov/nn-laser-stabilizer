from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config


class SACLoss:
    def __init__(
        self,
        gamma: float,
        log_alpha: nn.Parameter,
        target_entropy: float,
        auto_alpha: bool,
    ):
        self._gamma = gamma
        self.log_alpha = log_alpha
        self._target_entropy = target_entropy
        self._auto_alpha = auto_alpha

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @classmethod
    def from_config(cls, algorithm_config: Config) -> "SACLoss":
        gamma = float(algorithm_config.gamma)
        initial_alpha = float(algorithm_config.initial_alpha)
        auto_alpha = bool(algorithm_config.auto_alpha)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")

        log_alpha = nn.Parameter(torch.tensor(initial_alpha).log())

        # Heuristic: target_entropy = -action_dim
        # action_dim is inferred at learner build time; use a placeholder.
        # We'll set it properly via set_target_entropy.
        target_entropy = 0.0

        return cls(
            gamma=gamma,
            log_alpha=log_alpha,
            target_entropy=target_entropy,
            auto_alpha=auto_alpha,
        )

    def set_target_entropy(self, action_dim: int) -> None:
        self._target_entropy = -float(action_dim)

    def critic_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        current_q1 = output["current_q1"]
        current_q2 = output["current_q2"]
        target_q1 = output["target_q1"]
        target_q2 = output["target_q2"]
        next_log_prob = output["next_log_prob"]
        rewards = output["rewards"]
        dones = output["dones"]

        alpha = self.alpha.detach()

        with torch.no_grad():
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        return {
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
        }

    def actor_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        actor_q1 = output["actor_q1"]
        actor_q2 = output["actor_q2"]
        actor_log_prob = output["actor_log_prob"]

        alpha = self.alpha.detach()
        min_q = torch.min(actor_q1, actor_q2)
        actor_loss = (alpha * actor_log_prob - min_q).mean()

        return {
            "actor_loss": actor_loss,
        }

    def alpha_loss(self, output: dict[str, Any]) -> dict[str, Tensor]:
        actor_log_prob = output["actor_log_prob"]

        alpha_loss = -(self.log_alpha * (actor_log_prob + self._target_entropy).detach()).mean()

        return {
            "alpha_loss": alpha_loss,
        }

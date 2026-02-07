"""Behavioral cloning loss: MSE between policy(obs) and dataset actions."""

import torch
import torch.nn.functional as F

from nn_laser_stabilizer.rl.model.actor import Actor


class BCLoss:
    def __init__(self, actor: Actor):
        self._actor = actor

    def actor_loss(
        self,
        observations: torch.Tensor,
        dataset_actions: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between π(observations) and dataset_actions."""
        actions, _ = self._actor(observations)
        return F.mse_loss(actions, dataset_actions)

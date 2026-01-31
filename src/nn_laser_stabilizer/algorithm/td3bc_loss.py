from nn_laser_stabilizer.algorithm.td3_loss import TD3Loss
from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.model.critic import Critic


import torch
import torch.nn.functional as F


class TD3BCLoss:
    EPSILON = 1e-8

    def __init__(
        self,
         actor: Actor,
        critic1: Critic,
        critic2: Critic,
        actor_target: Actor,
        critic1_target: Critic,
        critic2_target: Critic,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
        alpha: float
    ):
        self._td3_loss = TD3Loss(
            actor=actor,
            critic1=critic1,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            critic2=critic2,
            actor_target=actor_target,
            critic1_target=critic1_target,
            critic2_target=critic2_target,
        )
        self.alpha = alpha

        self._actor = actor
        self._critic1 = critic1

    def critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._td3_loss.critic_loss(observations, actions, rewards, next_observations, dones)

    def actor_loss(
        self,
        observations: torch.Tensor,
        dataset_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions, _ = self._actor(observations)
        q_value, _ = self._critic1(observations, actions)

        with torch.no_grad():
            q_dataset, _ = self._critic1(observations, dataset_actions)
            lambda_coef = 1.0 / (torch.abs(q_dataset).mean().item() + self.EPSILON)

        policy_term = -lambda_coef * q_value.mean()
        bc_term = self.alpha * F.mse_loss(actions, dataset_actions)
        return policy_term, bc_term
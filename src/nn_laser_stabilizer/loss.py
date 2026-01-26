from typing import Tuple, Union

import torch
import torch.nn.functional as F

from nn_laser_stabilizer.actor import Actor
from nn_laser_stabilizer.critic import Critic
from nn_laser_stabilizer.box import Box
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import LossType


class TD3Loss:
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        action_space: Box,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
    ):
        self._actor = actor
        # TODO: можно перейти к списку критиков
        self._critic1 = critic
        self._critic2 = critic.clone(reinitialize_weights=True)

        self._actor_target = self._actor.clone().requires_grad_(False)
        self._critic1_target = self._critic1.clone().requires_grad_(False)
        self._critic2_target = self._critic2.clone().requires_grad_(False)
        
        self._action_space = action_space
        self._gamma = gamma
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        
        self._min_action = action_space.low
        self._max_action = action_space.high
    
    @property
    def actor(self):
        return self._actor
    
    @property
    def critic1(self):
        return self._critic1
    
    @property
    def critic2(self):
        return self._critic2
    
    @property
    def actor_target(self):
        return self._actor_target
    
    @property
    def critic1_target(self):
        return self._critic1_target
    
    @property
    def critic2_target(self):
        return self._critic2_target
    
    def critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_actions, _ = self._actor_target(next_observations)
            noise = (torch.randn_like(next_actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (next_actions + noise).clamp(self._min_action, self._max_action)
         
            target_q1, _ = self._critic1_target(next_observations, next_actions)
            target_q2, _ = self._critic2_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())
        
        current_q1, _ = self._critic1(observations, actions)
        current_q2, _ = self._critic2(observations, actions)
        
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)
        return loss_q1, loss_q2
    
    def actor_loss(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        actions, _ = self._actor(observations)
        q_value, _ = self._critic1(observations, actions)
    
        actor_loss = -q_value.mean()
        return actor_loss


class TD3BCLoss:
    EPSILON = 1e-8
    
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        action_space: Box,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
        alpha: float,
    ):
        self._td3_loss = TD3Loss(
            actor=actor,
            critic=critic,
            action_space=action_space,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )
        self.alpha = alpha
    
    @property
    def actor(self):
        return self._td3_loss.actor
    
    @property
    def critic1(self):
        return self._td3_loss.critic1
    
    @property
    def critic2(self):
        return self._td3_loss.critic2
    
    @property
    def actor_target(self):
        return self._td3_loss.actor_target
    
    @property
    def critic1_target(self):
        return self._td3_loss.critic1_target
    
    @property
    def critic2_target(self):
        return self._td3_loss.critic2_target
    
    def critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._td3_loss.critic_loss(observations, actions, rewards, next_observations, dones)
    
    def actor_loss(
        self,
        observations: torch.Tensor,
        dataset_actions: torch.Tensor,
    ) -> torch.Tensor:
        actions, _ = self.actor(observations)
        q_value, _ = self.critic1(observations, actions)
    
        with torch.no_grad():
            q_dataset, _ = self.critic1(observations, dataset_actions)
            lambda_coef = 1.0 / (torch.abs(q_dataset).mean().item() + self.EPSILON)
        
        bc_term = self.alpha * F.mse_loss(actions, dataset_actions)
        actor_loss = -lambda_coef * q_value.mean() + bc_term
        return actor_loss


def make_loss_from_config(
    loss_config: Config,
    actor: Actor,
    critic: Critic,
    action_space: Box
) -> Union[TD3Loss, TD3BCLoss]:
    loss_type_str = loss_config.type
    loss_type = LossType.from_str(loss_type_str)
    
    if loss_type == LossType.TD3:
        return TD3Loss(
            actor=actor,
            critic=critic,
            action_space=action_space,
            gamma=loss_config.gamma,
            policy_noise=loss_config.policy_noise,
            noise_clip=loss_config.noise_clip,
        )
    elif loss_type == LossType.TD3BC:
        alpha = loss_config.alpha
        return TD3BCLoss(
            actor=actor,
            critic=critic,
            action_space=action_space,
            gamma=loss_config.gamma,
            policy_noise=loss_config.policy_noise,
            noise_clip=loss_config.noise_clip,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unhandled loss type: {loss_type}")
    
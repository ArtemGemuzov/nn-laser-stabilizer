from typing import Tuple

import torch
import torch.nn.functional as F

from nn_laser_stabilizer.actor import Actor
from nn_laser_stabilizer.critic import Critic
from nn_laser_stabilizer.box import Box


class TD3Loss:
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        action_space: Box,
        gamma: float = 0.99,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        self.actor = actor
        # TODO: можно перейти к списку критиков
        self.critic1 = critic
        self.critic2 = critic.clone(reinitialize_weights=True)

        self.actor_target = self.actor.clone().requires_grad_(False)
        self.critic1_target = self.critic1.clone().requires_grad_(False)
        self.critic2_target = self.critic2.clone().requires_grad_(False)
        
        self.action_space = action_space
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        self.min_action = action_space.low
        self.max_action = action_space.high
    
    def critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_actions = self.actor_target(next_observations)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(self.min_action, self.max_action)
         
            target_q1 = self.critic1_target(next_observations, next_actions)
            target_q2 = self.critic2_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = rewards + self.gamma * target_q * (1.0 - dones.float())
        
        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)
        
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)
        return loss_q1, loss_q2
    
    def actor_loss(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        actions = self.actor(observations)
        q_value = self.critic1(observations, actions)
    
        actor_loss = -q_value.mean()
        return actor_loss
    
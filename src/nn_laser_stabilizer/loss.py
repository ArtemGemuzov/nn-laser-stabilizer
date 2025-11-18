from typing import Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.critic import Critic


def make_target(network: nn.Module) -> nn.Module:
    target = copy.deepcopy(network)
    target.load_state_dict(network.state_dict())
    for param in target.parameters():
        param.requires_grad = False
    return target


class TD3Loss:
    def __init__(
        self,
        actor: Policy,
        critic: Critic,
        action_space,
        gamma: float = 0.99,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        self.actor = actor
        self.critic1 = critic
        self.critic2 = copy.deepcopy(critic)

        self.actor_target = make_target(actor)
        self.critic1_target = make_target(critic)
        self.critic2_target = make_target(critic)
        
        self.action_space = action_space
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        self.min_action = torch.tensor(action_space.low, dtype=torch.float32)
        self.max_action = torch.tensor(action_space.high, dtype=torch.float32)
    
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
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = next_actions + noise
         
            next_actions = torch.clamp(
                next_actions,
                min=self.min_action,
                max=self.max_action
            )
            
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
    
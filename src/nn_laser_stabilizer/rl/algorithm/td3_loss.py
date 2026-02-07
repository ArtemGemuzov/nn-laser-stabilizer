import torch
import torch.nn.functional as F

from nn_laser_stabilizer.rl.model.actor import Actor
from nn_laser_stabilizer.rl.model.critic import Critic


class TD3Loss:
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
    ):
        self._actor = actor
        self._critic1 = critic1
        self._critic2 = critic2

        self._actor_target = actor_target.requires_grad_(False)
        self._critic1_target = critic1_target.requires_grad_(False)
        self._critic2_target = critic2_target.requires_grad_(False)

        action_space = actor.action_space
        self._action_space = action_space
        self._gamma = gamma
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._min_action = action_space.low
        self._max_action = action_space.high

    def critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
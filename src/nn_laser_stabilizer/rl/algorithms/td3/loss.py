import torch
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent


class TD3Loss:
    def __init__(
        self,
        agent: TD3Agent,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
    ):
        self._agent = agent
        self._gamma = gamma
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

    @classmethod
    def from_config(cls, algorithm_config: Config, agent: TD3Agent) -> "TD3Loss":
        gamma = float(algorithm_config.gamma)
        policy_noise = float(algorithm_config.policy_noise)
        noise_clip = float(algorithm_config.noise_clip)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")

        return cls(
            agent=agent,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )

    def critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        agent = self._agent

        with torch.no_grad():
            target_output = agent.actor_target(next_obs)
            noise = (torch.randn_like(target_output.action) * self._policy_noise).clamp(
                -self._noise_clip, self._noise_clip
            )
            next_actions = torch.clamp(
                target_output.action + noise,
                agent.action_space.low,
                agent.action_space.high,
            )

            target_q1 = agent.critic1_target(next_obs, next_actions).q_value
            target_q2 = agent.critic2_target(next_obs, next_actions).q_value
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        current_q1 = agent.critic1(obs, actions).q_value
        current_q2 = agent.critic2(obs, actions).q_value

        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        return loss_q1, loss_q2

    def actor_loss(self, obs: Tensor) -> Tensor:
        agent = self._agent
        output = agent.actor(obs)
        q_value = agent.critic1(obs, output.action).q_value
        return -q_value.mean()

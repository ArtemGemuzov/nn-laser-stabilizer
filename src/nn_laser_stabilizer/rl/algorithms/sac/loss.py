import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.sac.agent import SACAgent
from nn_laser_stabilizer.rl.policy.gaussian import (
    tanh_squash,
    gaussian_log_prob,
    LOG_STD_MIN,
    LOG_STD_MAX,
)


class SACLoss:
    def __init__(
        self,
        agent: SACAgent,
        gamma: float,
        log_alpha: nn.Parameter,
        target_entropy: float,
    ):
        self._agent = agent
        self._gamma = gamma
        self.log_alpha = log_alpha
        self._target_entropy = target_entropy

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @classmethod
    def from_config(cls, algorithm_config: Config, agent: SACAgent) -> "SACLoss":
        gamma = float(algorithm_config.gamma)
        initial_alpha = float(algorithm_config.initial_alpha)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")

        log_alpha = nn.Parameter(torch.tensor(initial_alpha).log())
        target_entropy = -float(agent.action_space.dim)

        return cls(
            agent=agent,
            gamma=gamma,
            log_alpha=log_alpha,
            target_entropy=target_entropy,
        )

    def _sample_actions(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        raw = self._agent.actor_net(observations)
        mean, log_std = raw.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action_space = self._agent.action_space
        actions = tanh_squash(x_t, action_space.low, action_space.high)
        action_scale = (action_space.high - action_space.low) / 2.0
        log_prob = gaussian_log_prob(normal, x_t, action_scale)
        return actions, log_prob

    def critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_obs: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        agent = self._agent
        alpha = self.alpha.detach()

        current_q1, _ = agent.critic1(obs, actions)
        current_q2, _ = agent.critic2(obs, actions)

        with torch.no_grad():
            next_actions, next_log_prob = self._sample_actions(next_obs)
            target_q1, _ = agent.critic1_target(next_obs, next_actions)
            target_q2, _ = agent.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        return loss_q1, loss_q2

    def actor_loss(self, obs: Tensor) -> dict[str, Tensor]:
        agent = self._agent
        alpha = self.alpha.detach()

        actor_actions, actor_log_prob = self._sample_actions(obs)
        actor_q1, _ = agent.critic1(obs, actor_actions)
        actor_q2, _ = agent.critic2(obs, actor_actions)

        min_q = torch.min(actor_q1, actor_q2)
        actor_loss = (alpha * actor_log_prob - min_q).mean()

        return {
            "actor_loss": actor_loss,
            "actor_log_prob": actor_log_prob,
        }

    def alpha_loss(self, actor_log_prob: Tensor) -> Tensor:
        return -(self.log_alpha * (actor_log_prob + self._target_entropy).detach()).mean()

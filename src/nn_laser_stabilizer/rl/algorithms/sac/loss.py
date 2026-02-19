import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.sac.agent import SACAgent


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

        current_q1 = agent.critic1(obs, actions).q_value
        current_q2 = agent.critic2(obs, actions).q_value

        with torch.no_grad():
            next_output = agent.actor(next_obs)
            target_q1 = agent.critic1_target(next_obs, next_output.action).q_value
            target_q2 = agent.critic2_target(next_obs, next_output.action).q_value
            target_q = torch.min(target_q1, target_q2) - alpha * next_output.log_prob
            target_q = rewards + self._gamma * target_q * (1.0 - dones.float())

        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        return loss_q1, loss_q2

    def actor_loss(self, obs: Tensor) -> dict[str, Tensor]:
        agent = self._agent
        alpha = self.alpha.detach()

        output = agent.actor(obs)
        actor_q1 = agent.critic1(obs, output.action).q_value
        actor_q2 = agent.critic2(obs, output.action).q_value

        min_q = torch.min(actor_q1, actor_q2)
        actor_loss = (alpha * output.log_prob - min_q).mean()

        return {
            "actor_loss": actor_loss,
            "actor_log_prob": output.log_prob,
        }

    def alpha_loss(self, actor_log_prob: Tensor) -> Tensor:
        return -(self.log_alpha * (actor_log_prob + self._target_entropy).detach()).mean()

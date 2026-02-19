import torch
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.algorithms.td3.loss import TD3Loss


class TD3BCLoss(TD3Loss):
    EPSILON = 1e-8

    def __init__(
        self,
        agent: TD3Agent,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
        alpha: float,
    ):
        super().__init__(agent=agent, gamma=gamma, policy_noise=policy_noise, noise_clip=noise_clip)
        self._alpha = alpha

    @classmethod
    def from_config(cls, algorithm_config: Config, agent: TD3Agent) -> "TD3BCLoss":
        gamma = float(algorithm_config.gamma)
        policy_noise = float(algorithm_config.policy_noise)
        noise_clip = float(algorithm_config.noise_clip)
        alpha = float(algorithm_config.alpha)

        if gamma <= 0.0:
            raise ValueError("algorithm.gamma must be > 0")

        return cls(
            agent=agent,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
        )

    def actor_loss(self, obs: Tensor, actions: Tensor) -> dict[str, Tensor]:
        agent = self._agent
        output = agent.actor(obs)
        q_value = agent.critic1(obs, output.action).q_value

        with torch.no_grad():
            q_dataset = agent.critic1(obs, actions).q_value
            lambda_coef = 1.0 / (torch.abs(q_dataset).mean().item() + self.EPSILON)

        td3_term = -lambda_coef * q_value.mean()
        bc_term = self._alpha * F.mse_loss(output.action, actions)
        actor_loss = td3_term + bc_term

        return {
            "actor_loss": actor_loss,
            "actor_td3_term": td3_term,
            "actor_bc_term": bc_term,
        }

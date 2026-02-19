import torch
import torch.nn.functional as F
from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.envs.spaces.box import Box


class TD3BCAgent(TD3Agent):
    EPSILON = 1e-8

    def __init__(self, *args, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        observation_space: Box,
        action_space: Box,
    ) -> "TD3BCAgent":
        base = TD3Agent.from_config(algorithm_config, observation_space, action_space)
        alpha = float(algorithm_config.alpha)

        return cls(
            actor=base._actor,
            actor_target=base._actor_target,
            critic1=base._critic1,
            critic2=base._critic2,
            critic1_target=base._critic1_target,
            critic2_target=base._critic2_target,
            action_space=base._action_space,
            actor_optimizer=base._actor_optimizer,
            critic_optimizer=base._critic_optimizer,
            soft_updater=base._soft_updater,
            gamma=base._gamma,
            policy_noise=base._policy_noise,
            noise_clip=base._noise_clip,
            policy_freq=base._policy_freq,
            alpha=alpha,
        )

    def _actor_loss(self, obs: Tensor, actions: Tensor) -> Tensor:
        output = self._actor(obs)
        q_value = self._critic1(obs, output.action).q_value

        with torch.no_grad():
            q_dataset = self._critic1(obs, actions).q_value
            lambda_coef = 1.0 / (torch.abs(q_dataset).mean().item() + self.EPSILON)

        td3_term = -lambda_coef * q_value.mean()
        bc_term = self._alpha * F.mse_loss(output.action, actions)
        return td3_term + bc_term

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._critic_loss(obs, actions, rewards, next_obs, dones)
        self._critic_optimizer.step(loss_q1 + loss_q2)

        metrics: dict[str, float] = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
        }

        self._step += 1
        if self._step % self._policy_freq == 0:
            actor_loss = self._actor_loss(obs, actions)
            self._actor_optimizer.step(actor_loss)
            self._soft_updater.update()
            metrics["actor_loss"] = actor_loss.item()

        return metrics

from typing import Optional


from torch import Tensor


from nn_laser_stabilizer.algorithm.td3bc_loss import TD3BCLoss
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.model.critic import Critic
from nn_laser_stabilizer.optimizer import SoftUpdater
from nn_laser_stabilizer.algorithm.utils import OptimizerFactory, build_soft_update_pairs


class TD3BCUpdater:
    def __init__(
        self,
        updater_config: Config,
        actor: Actor,
        critic: Critic,
        actor_optimizer_factory: OptimizerFactory,
        critic_optimizer_factory: OptimizerFactory,
    ) -> None:
        self._policy_freq: int = int(updater_config.policy_freq)
        self._step: int = 0

        self._actor = actor
        self._critic1 = critic
        self._critic2 = critic.clone(reinitialize_weights=True)

        self._actor_target = self._actor.clone().requires_grad_(False)
        self._critic1_target = self._critic1.clone().requires_grad_(False)
        self._critic2_target = self._critic2.clone().requires_grad_(False)

        self._loss_module = TD3BCLoss(
            actor=self._actor,
            critic1=self._critic1,
            critic2=self._critic2,
            actor_target=self._actor_target,
            critic1_target=self._critic1_target,
            critic2_target=self._critic2_target,
            gamma=updater_config.gamma,
            policy_noise=updater_config.policy_noise,
            noise_clip=updater_config.noise_clip,
            alpha=updater_config.alpha,
        )

        self._actor_optimizer = actor_optimizer_factory(self._actor.parameters())
        self._critic_optimizer = critic_optimizer_factory(
            list(self._critic1.parameters()) + list(self._critic2.parameters())
        )
        self._soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (self._actor_target, self._actor),
                    (self._critic1_target, self._critic1),
                    (self._critic2_target, self._critic2),
                )
            ),
            tau=updater_config.tau,
        )

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

    def _should_update_actor_and_target(self) -> bool:
        self._step += 1
        return (self._step % self._policy_freq) == 0

    def update_step(self, batch: tuple[Tensor, ...]) -> tuple[float, float, Optional[float]]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._loss_module.critic_loss(
            obs, actions, rewards, next_obs, dones
        )
        self._critic_optimizer.step((loss_q1 + loss_q2).sum())

        actor_loss_value: Optional[Tensor] = None
        if self._should_update_actor_and_target():
            actor_loss_value = self._loss_module.actor_loss(
                obs,
                dataset_actions=actions,
            )
            self._actor_optimizer.step(actor_loss_value)
            self._soft_updater.update()

        return (
            loss_q1.item(),
            loss_q2.item(),
            actor_loss_value.item() if actor_loss_value is not None else None,
        )
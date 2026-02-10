from pathlib import Path

from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.agents.td3bc_loss import TD3BCLoss
from nn_laser_stabilizer.rl.model.actor import Actor
from nn_laser_stabilizer.rl.model.critic import Critic
from nn_laser_stabilizer.rl.agents.optimizer import SoftUpdater
from nn_laser_stabilizer.rl.agents.utils import OptimizerFactory, build_soft_update_pairs


class TD3BCUpdater:
    ACTOR_FILENAME = "actor.pth"
    CRITIC1_FILENAME = "critic1.pth"
    CRITIC2_FILENAME = "critic2.pth"
    ACTOR_TARGET_FILENAME = "actor_target.pth"
    CRITIC1_TARGET_FILENAME = "critic1_target.pth"
    CRITIC2_TARGET_FILENAME = "critic2_target.pth"

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        actor_optimizer_factory: OptimizerFactory,
        critic_optimizer_factory: OptimizerFactory,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
        alpha: float,
        tau: float,
        policy_freq: int,
    ):
        self._policy_freq: int = int(policy_freq)
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
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
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
            tau=tau,
        )

    @classmethod
    def from_config(
        cls,
        updater_config: Config,
        *,
        actor: Actor,
        critic: Critic,
        actor_optimizer_factory: OptimizerFactory,
        critic_optimizer_factory: OptimizerFactory,
    ) -> "TD3BCUpdater":
        gamma = float(updater_config.gamma)
        policy_noise = float(updater_config.policy_noise)
        noise_clip = float(updater_config.noise_clip)
        alpha = float(updater_config.alpha)
        tau = float(updater_config.tau)
        policy_freq = int(updater_config.policy_freq)

        if gamma <= 0.0:
            raise ValueError("updater.gamma must be > 0")
        if tau <= 0.0:
            raise ValueError("updater.tau must be > 0")
        if policy_freq <= 0:
            raise ValueError("updater.policy_freq must be > 0")

        return cls(
            actor=actor,
            critic=critic,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            tau=tau,
            policy_freq=policy_freq,
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

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self._actor.save(models_dir / self.ACTOR_FILENAME)
        self._critic1.save(models_dir / self.CRITIC1_FILENAME)
        self._critic2.save(models_dir / self.CRITIC2_FILENAME)
        self._actor_target.save(models_dir / self.ACTOR_TARGET_FILENAME)
        self._critic1_target.save(models_dir / self.CRITIC1_TARGET_FILENAME)
        self._critic2_target.save(models_dir / self.CRITIC2_TARGET_FILENAME)

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._loss_module.critic_loss(
            obs, actions, rewards, next_obs, dones
        )
        self._critic_optimizer.step((loss_q1 + loss_q2).sum())

        metrics: dict[str, float] = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
        }
        if self._should_update_actor_and_target():
            td3_term, bc_term = self._loss_module.actor_loss(
                obs,
                dataset_actions=actions,
            )
            actor_loss_value = td3_term + bc_term
            self._actor_optimizer.step(actor_loss_value)
            self._soft_updater.update()
            metrics["actor_loss"] = actor_loss_value.item()
            metrics["actor_td3_term"] = td3_term.item()
            metrics["actor_bc_term"] = bc_term.item()
        return metrics
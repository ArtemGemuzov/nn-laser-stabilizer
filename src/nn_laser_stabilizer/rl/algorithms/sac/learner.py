from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Learner
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs
from nn_laser_stabilizer.rl.algorithms.sac.agent import SACAgent
from nn_laser_stabilizer.rl.algorithms.sac.loss import SACLoss


class SACLearner(Learner):
    def __init__(
        self,
        agent: SACAgent,
        loss: SACLoss,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        alpha_optimizer: Optimizer,
        soft_updater: SoftUpdater,
    ):
        self._agent = agent
        self._loss = loss
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._alpha_optimizer = alpha_optimizer
        self._soft_updater = soft_updater

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        agent: SACAgent,
        loss: SACLoss,
    ) -> "SACLearner":
        tau = float(algorithm_config.tau)
        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")

        actor_lr = float(algorithm_config.actor.optimizer.lr)
        critic_lr = float(algorithm_config.critic.optimizer.lr)

        actor_optimizer = Optimizer(agent.actor.parameters(), lr=actor_lr)
        critic_optimizer = Optimizer(
            list(agent.critic1.parameters()) + list(agent.critic2.parameters()),
            lr=critic_lr,
        )

        alpha_lr = float(algorithm_config.alpha_optimizer.lr)
        alpha_optimizer = Optimizer([loss.log_alpha], lr=alpha_lr)

        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (agent.critic1_target, agent.critic1),
                    (agent.critic2_target, agent.critic2),
                )
            ),
            tau=tau,
        )

        return cls(
            agent=agent,
            loss=loss,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            soft_updater=soft_updater,
        )

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._loss.critic_loss(obs, actions, rewards, next_obs, dones)
        self._critic_optimizer.step(loss_q1 + loss_q2)

        actor_result = self._loss.actor_loss(obs)
        self._actor_optimizer.step(actor_result["actor_loss"])

        metrics: dict[str, float] = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "actor_loss": actor_result["actor_loss"].item(),
        }

        alpha_loss = self._loss.alpha_loss(actor_result["actor_log_prob"])
        self._alpha_optimizer.step(alpha_loss)
        metrics["alpha_loss"] = alpha_loss.item()

        metrics["alpha"] = self._loss.alpha.item()

        self._soft_updater.update()

        return metrics

from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Learner
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs
from nn_laser_stabilizer.rl.algorithms.td3.agent import TD3Agent
from nn_laser_stabilizer.rl.algorithms.td3.loss import TD3Loss


class TD3Learner(Learner):
    def __init__(
        self,
        agent: TD3Agent,
        loss: TD3Loss,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        soft_updater: SoftUpdater,
        policy_freq: int,
    ):
        self._agent = agent
        self._loss = loss
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._soft_updater = soft_updater
        self._policy_freq = policy_freq
        self._step = 0

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        agent: TD3Agent,
        loss: TD3Loss,
    ) -> "TD3Learner":
        tau = float(algorithm_config.tau)
        policy_freq = int(algorithm_config.policy_freq)

        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")
        if policy_freq <= 0:
            raise ValueError("algorithm.policy_freq must be > 0")

        actor_lr = float(algorithm_config.actor.optimizer.lr)
        critic_lr = float(algorithm_config.critic.optimizer.lr)

        actor_optimizer = Optimizer(agent.actor.parameters(), lr=actor_lr)
        critic_optimizer = Optimizer(
            list(agent.critic1.parameters()) + list(agent.critic2.parameters()),
            lr=critic_lr,
        )

        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (agent.actor_target, agent.actor),
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
            soft_updater=soft_updater,
            policy_freq=policy_freq,
        )

    def _should_update_actor_and_target(self) -> bool:
        self._step += 1
        return (self._step % self._policy_freq) == 0

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        loss_q1, loss_q2 = self._loss.critic_loss(obs, actions, rewards, next_obs, dones)
        self._critic_optimizer.step(loss_q1 + loss_q2)

        metrics: dict[str, float] = {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
        }

        if self._should_update_actor_and_target():
            actor_loss = self._loss.actor_loss(obs)
            self._actor_optimizer.step(actor_loss)
            self._soft_updater.update()

            metrics["actor_loss"] = actor_loss.item()
        return metrics

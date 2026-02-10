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
        alpha_optimizer: Optimizer | None,
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

        actor_optimizer = Optimizer(agent._actor_net.parameters(), lr=actor_lr)
        critic_optimizer = Optimizer(
            list(agent._critic1.parameters()) + list(agent._critic2.parameters()),
            lr=critic_lr,
        )

        # Alpha optimizer (only if auto_alpha)
        alpha_optimizer = None
        if loss._auto_alpha:
            alpha_lr = float(algorithm_config.alpha_optimizer.lr)
            alpha_optimizer = Optimizer([loss.log_alpha], lr=alpha_lr)

        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (agent._critic1_target, agent._critic1),
                    (agent._critic2_target, agent._critic2),
                )
            ),
            tau=tau,
        )

        # Set target entropy from action dim
        loss.set_target_entropy(agent._action_space.dim)

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

        output = self._agent.forward_train(obs, actions, rewards, next_obs, dones)

        # --- critic update ---
        critic_losses = self._loss.critic_loss(output)
        total_critic_loss = critic_losses["loss_q1"] + critic_losses["loss_q2"]
        self._critic_optimizer.step(total_critic_loss)

        # --- actor update (every step for SAC) ---
        actor_losses = self._loss.actor_loss(output)
        self._actor_optimizer.step(actor_losses["actor_loss"])

        metrics: dict[str, float] = {
            "loss_q1": critic_losses["loss_q1"].item(),
            "loss_q2": critic_losses["loss_q2"].item(),
            "actor_loss": actor_losses["actor_loss"].item(),
        }

        # --- alpha update ---
        if self._alpha_optimizer is not None:
            alpha_losses = self._loss.alpha_loss(output)
            self._alpha_optimizer.step(alpha_losses["alpha_loss"])
            metrics["alpha_loss"] = alpha_losses["alpha_loss"].item()
            metrics["alpha"] = self._loss.alpha.item()

        # --- soft update (critics only, no target actor) ---
        self._soft_updater.update()

        return metrics

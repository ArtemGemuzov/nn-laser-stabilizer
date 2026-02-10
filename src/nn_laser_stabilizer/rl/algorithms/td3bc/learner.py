from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.td3.learner import TD3Learner
from nn_laser_stabilizer.rl.algorithms.td3bc.agent import TD3BCAgent
from nn_laser_stabilizer.rl.algorithms.td3bc.loss import TD3BCLoss
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.rl.algorithms.utils import build_soft_update_pairs


class TD3BCLearner(TD3Learner):
    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        agent: TD3BCAgent,
        loss: TD3BCLoss,
    ) -> "TD3BCLearner":
        tau = float(algorithm_config.tau)
        policy_freq = int(algorithm_config.policy_freq)

        if tau <= 0.0:
            raise ValueError("algorithm.tau must be > 0")
        if policy_freq <= 0:
            raise ValueError("algorithm.policy_freq must be > 0")

        actor_lr = float(algorithm_config.actor.optimizer.lr)
        critic_lr = float(algorithm_config.critic.optimizer.lr)

        actor_optimizer = Optimizer(agent._actor.parameters(), lr=actor_lr)
        critic_optimizer = Optimizer(
            list(agent._critic1.parameters()) + list(agent._critic2.parameters()),
            lr=critic_lr,
        )

        soft_updater = SoftUpdater(
            pairs=build_soft_update_pairs(
                module_pairs=(
                    (agent._actor_target, agent._actor),
                    (agent._critic1_target, agent._critic1),
                    (agent._critic2_target, agent._critic2),
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

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch

        output = self._agent.forward_train(obs, actions, rewards, next_obs, dones)

        # --- critic update ---
        critic_losses = self._loss.critic_loss(output)
        total_critic_loss = critic_losses["loss_q1"] + critic_losses["loss_q2"]
        self._critic_optimizer.step(total_critic_loss)

        metrics: dict[str, float] = {
            "loss_q1": critic_losses["loss_q1"].item(),
            "loss_q2": critic_losses["loss_q2"].item(),
        }

        # --- actor update ---
        if self._should_update_actor_and_target():
            actor_losses = self._loss.actor_loss(output)
            self._actor_optimizer.step(actor_losses["actor_loss"])
            self._soft_updater.update()
            metrics["actor_loss"] = actor_losses["actor_loss"].item()
            metrics["actor_td3_term"] = actor_losses["actor_td3_term"].item()
            metrics["actor_bc_term"] = actor_losses["actor_bc_term"].item()

        return metrics

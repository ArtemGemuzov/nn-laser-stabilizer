from torch import Tensor

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.rl.algorithms.base import Learner
from nn_laser_stabilizer.rl.algorithms.optimizer import Optimizer
from nn_laser_stabilizer.rl.algorithms.bc.agent import BCAgent
from nn_laser_stabilizer.rl.algorithms.bc.loss import BCLoss


class BCLearner(Learner):
    def __init__(
        self,
        agent: BCAgent,
        loss: BCLoss,
        actor_optimizer: Optimizer,
    ):
        self._agent = agent
        self._loss = loss
        self._actor_optimizer = actor_optimizer

    @classmethod
    def from_config(
        cls,
        algorithm_config: Config,
        agent: BCAgent,
        loss: BCLoss,
    ) -> "BCLearner":
        actor_lr = float(algorithm_config.actor.optimizer.lr)
        actor_optimizer = Optimizer(agent._actor.parameters(), lr=actor_lr)

        return cls(
            agent=agent,
            loss=loss,
            actor_optimizer=actor_optimizer,
        )

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, *_ = batch

        output = self._agent.forward_train(obs, actions, Tensor(), Tensor(), Tensor())

        actor_losses = self._loss.actor_loss(output)
        self._actor_optimizer.step(actor_losses["actor_loss"])

        return {"actor_loss": actor_losses["actor_loss"].item()}

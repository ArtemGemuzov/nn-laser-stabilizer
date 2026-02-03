from pathlib import Path

from torch import Tensor

from nn_laser_stabilizer.algorithm.bc_loss import BCLoss
from nn_laser_stabilizer.algorithm.utils import OptimizerFactory
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.model.critic import Critic


class BCUpdater:
    ACTOR_FILENAME = "actor.pth"

    def __init__(
        self,
        actor: Actor,
        actor_optimizer_factory: OptimizerFactory,
    ):
        self._actor = actor
        self._loss_module = BCLoss(actor=actor)
        self._actor_optimizer = actor_optimizer_factory(self._actor.parameters())

    @classmethod
    def from_config(
        cls,
        updater_config: Config,
        *,
        actor: Actor,
        critic: Critic | None,
        actor_optimizer_factory: OptimizerFactory,
        critic_optimizer_factory: OptimizerFactory | None = None,
    ) -> "BCUpdater":
        return cls(
            actor=actor,
            actor_optimizer_factory=actor_optimizer_factory,
        )

    @property
    def actor(self) -> Actor:
        return self._actor

    @property
    def critic1(self) -> None:
        return None

    @property
    def critic2(self) -> None:
        return None

    @property
    def actor_target(self) -> None:
        return None

    @property
    def critic1_target(self) -> None:
        return None

    @property
    def critic2_target(self) -> None:
        return None

    def save_models(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self._actor.save(models_dir / self.ACTOR_FILENAME)

    def update_step(self, batch: tuple[Tensor, ...]) -> dict[str, float]:
        obs, actions, *_ = batch
        actor_loss = self._loss_module.actor_loss(obs, dataset_actions=actions)
        self._actor_optimizer.step(actor_loss)
        return {"actor_loss": actor_loss.item()}

from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.config.types import UpdaterType
from nn_laser_stabilizer.rl.agents.bc_updater import BCUpdater
from nn_laser_stabilizer.rl.agents.td3_updater import TD3Updater
from nn_laser_stabilizer.rl.agents.td3bc_updater import TD3BCUpdater
from nn_laser_stabilizer.rl.agents.utils import OptimizerFactory
from nn_laser_stabilizer.rl.model.actor import Actor
from nn_laser_stabilizer.rl.model.critic import Critic


def make_updater_from_config(
    updater_config: Config,
    actor: Actor,
    critic: Critic,
    actor_optimizer_factory: OptimizerFactory,
    critic_optimizer_factory: OptimizerFactory,
) -> TD3Updater | TD3BCUpdater | BCUpdater:
    loss_type = UpdaterType.from_str(updater_config.type)
    if loss_type == UpdaterType.TD3:
        return TD3Updater.from_config(
            updater_config=updater_config,
            actor=actor,
            critic=critic,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
        )
    elif loss_type == UpdaterType.TD3BC:
        return TD3BCUpdater.from_config(
            updater_config=updater_config,
            actor=actor,
            critic=critic,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
        )
    elif loss_type == UpdaterType.BC:
        return BCUpdater.from_config(
            updater_config=updater_config,
            actor=actor,
            critic=critic,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
        )
    else:
        raise ValueError(f"Unhandled updater type: {loss_type}")
from typing import Callable, Iterable

import torch
import torch.nn as nn

from nn_laser_stabilizer.algorithm.td3_updater import TD3Updater
from nn_laser_stabilizer.algorithm.td3bc_updater import TD3BCUpdater
from nn_laser_stabilizer.model.actor import Actor
from nn_laser_stabilizer.config.config import Config
from nn_laser_stabilizer.model.critic import Critic
from nn_laser_stabilizer.config.types import UpdaterType
from nn_laser_stabilizer.optimizer import Optimizer


OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], Optimizer]


def make_updater_from_config(
    updater_config: Config,
    actor: Actor,
    critic: Critic,
    actor_optimizer_factory: OptimizerFactory,
    critic_optimizer_factory: OptimizerFactory,
) -> TD3Updater | TD3BCUpdater:
    loss_type = UpdaterType.from_str(updater_config.type)

    if loss_type == UpdaterType.TD3:
        return TD3Updater(
            updater_config=updater_config,
            actor=actor,
            critic=critic,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
        )
    elif loss_type == UpdaterType.TD3BC:
        return TD3BCUpdater(
            updater_config=updater_config,
            actor=actor,
            critic=critic,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
        )
    else:
        raise ValueError(f"Unhandled updater type: {loss_type}")


def build_soft_update_pairs(
    *,
    module_pairs: Iterable[tuple[nn.Module, nn.Module]],
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for tgt, src in module_pairs:
        for t_param, s_param in zip(tgt.parameters(), src.parameters()):
            pairs.append((t_param, s_param))
    return pairs

from typing import Tuple, Optional

from torch import Tensor

from nn_laser_stabilizer.loss import TD3Loss, TD3BCLoss
from nn_laser_stabilizer.optimizer import Optimizer, SoftUpdater


def td3_train_step(
    batch: Tuple[Tensor, ...],
    loss_module: TD3Loss,
    critic_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    soft_updater: SoftUpdater,
    update_actor_and_target: bool = False,
) -> Tuple[float, float, Optional[float]]:
    obs, actions, rewards, next_obs, dones = batch
    
    loss_q1, loss_q2 = loss_module.critic_loss(obs, actions, rewards, next_obs, dones)
    critic_optimizer.step((loss_q1 + loss_q2).sum())
    
    actor_loss = None
    if update_actor_and_target:
        actor_loss = loss_module.actor_loss(obs)
        actor_optimizer.step(actor_loss)
        soft_updater.update()
    
    return loss_q1.item(), loss_q2.item(), actor_loss.item() if actor_loss is not None else None


def td3bc_train_step(
    batch: Tuple[Tensor, ...],
    loss_module: TD3BCLoss,
    critic_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    soft_updater: SoftUpdater,
    update_actor_and_target: bool = False,
) -> Tuple[float, float, Optional[float]]:
    obs, actions, rewards, next_obs, dones = batch
    
    loss_q1, loss_q2 = loss_module.critic_loss(obs, actions, rewards, next_obs, dones)
    critic_optimizer.step((loss_q1 + loss_q2).sum())
    
    actor_loss = None
    if update_actor_and_target:
        actor_loss = loss_module.actor_loss(obs, dataset_actions=actions)
        actor_optimizer.step(actor_loss)
        soft_updater.update()
    
    return loss_q1.item(), loss_q2.item(), actor_loss.item() if actor_loss is not None else None
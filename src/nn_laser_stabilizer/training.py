from typing import Tuple, Optional
import torch
import torch.optim as optim

from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.utils import SoftUpdater


def td3_train_step(
    batch: Tuple[torch.Tensor, ...],
    loss_module: TD3Loss,
    critic_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    soft_updater: SoftUpdater,
    update_actor_and_target: bool = False,
) -> Tuple[float, float, Optional[float]]:
    obs, actions, rewards, next_obs, dones = batch
    
    critic_optimizer.zero_grad(set_to_none=True)
    loss_q1, loss_q2 = loss_module.critic_loss(obs, actions, rewards, next_obs, dones)
    (loss_q1 + loss_q2).sum().backward()
    critic_optimizer.step()
    
    actor_loss = None
    if update_actor_and_target:
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = loss_module.actor_loss(obs)
        actor_loss.backward()
        actor_optimizer.step()
        
        soft_updater.update()
    
    return loss_q1.item(), loss_q2.item(), actor_loss.item() if actor_loss is not None else None
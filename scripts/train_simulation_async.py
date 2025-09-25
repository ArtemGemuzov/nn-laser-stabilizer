import os
from collections import deque

from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.logging.utils import (
    set_seeds
)
from nn_laser_stabilizer.agents.td3 import (
    make_td3_agent,
    add_exploration,
    make_loss_module,
    make_optimizers,
    make_target_updater,
    train_step,
    warmup,
    warmup_from_specs
)
from nn_laser_stabilizer.envs.utils import make_gym_env, add_logger_to_env, make_specs

from nn_laser_stabilizer.data.utils import make_buffer, make_async_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir

from logging import getLogger
logger = getLogger(__name__)

CONFIG_NAME = "train_simulation"

@hydra.main(config_path=find_configs_dir(), config_name=CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    def make_env_fn():
        return make_gym_env(config)
    
    eval_env = make_env_fn()
    action_spec = eval_env.action_spec_unbatched
    observation_spec = eval_env.observation_spec_unbatched["observation"]

    actor, qvalue = make_td3_agent(config, observation_spec, action_spec)

    warmup_from_specs(observation_spec, action_spec, actor, qvalue)

    buffer = make_buffer(config)
    collector = make_async_collector(config, make_env_fn, actor, buffer)

    loss_module = make_loss_module(config, actor, qvalue, action_spec)
    optimizer_actor, optimizer_critic = make_optimizers(config, loss_module)
    target_net_updater = make_target_updater(config, loss_module)

    train_config = config.train

    total_train_steps = 0

    recent_qvalue_losses = deque(maxlen=train_config.update_to_data)
    recent_actor_losses = deque(maxlen=train_config.update_to_data // train_config.update_actor_freq)

    try:
        logger.info("Training started")
        collector.start()
        
        while total_train_steps < train_config.total_train_steps:
            if len(buffer) > train_config.batch_size:
                for i in range(train_config.update_to_data):
                    batch = buffer.sample(train_config.batch_size)
                    
                    update_actor = i % train_config.update_actor_freq == 0
                    loss_qvalue_val, loss_actor_val = train_step(
                        batch, loss_module, optimizer_actor, optimizer_critic, 
                        target_net_updater, update_actor
                    )
                    
                    recent_qvalue_losses.append(loss_qvalue_val)
                    if loss_actor_val is not None:
                        recent_actor_losses.append(loss_actor_val)

                collector.update_policy_weights_()

                # TODO Попробовать вернуть exploration module
                
                avg_qvalue_loss = sum(recent_qvalue_losses) / len(recent_qvalue_losses)
                logger.info(f"Loss/Q-function = {avg_qvalue_loss} on step {total_train_steps}")

                avg_actor_loss = sum(recent_actor_losses) / len(recent_actor_losses)
                logger.info(f"Loss/Actor = {avg_actor_loss} on step {total_train_steps}")

                total_train_steps += 1

                tensordict_data = eval_env.rollout(max_steps=config.data.frames_per_batch, policy=actor, break_when_any_done=False)

                rewards = tensordict_data.get(("next", "reward"))
                steps = tensordict_data.get(("next", "step_count"))

                avg_reward = rewards.mean().item()
                logger.info(f"[Environment] Average reward per step = {avg_reward:.4f}")

                max_episode_length = steps.max().item()
                logger.info(f"[Environment] Max episode length = {max_episode_length}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception as ex:
        logger.warning(f"Error while training: {ex}")

    finally:
        logger.info("Training finished")
        
        collector.async_shutdown()


if __name__ == "__main__":
    main()

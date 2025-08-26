import os
from collections import deque

from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.train_utils import (
    set_seeds
)
from nn_laser_stabilizer.agents.td3 import (
    make_td3_agent,
    add_exploration,
    make_loss_module,
    make_optimizers,
    make_target_updater,
    train_step,
    warmup
)
from nn_laser_stabilizer.envs.utils import make_env, add_logger_to_env

from nn_laser_stabilizer.data.utils import make_buffer, make_async_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir, DEFAULE_CONFIG_NAME

from logging import getLogger
logger = getLogger(__name__)

from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path=find_configs_dir(), config_name=DEFAULE_CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    env_log_dir = os.path.join(hydra_output_dir, "env_logs")
    os.makedirs(env_log_dir, exist_ok=True)

    def make_env_fn():
        env = make_env(config)
        env = add_logger_to_env(env, env_log_dir)
        return env

    train_log_dir = os.path.join(hydra_output_dir, "train_logs")
    os.makedirs(train_log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=train_log_dir)

    temp_env = make_env(config)
    action_spec = temp_env.action_spec_unbatched
    observation_spec = temp_env.observation_spec_unbatched["observation"]

    actor, qvalue = make_td3_agent(config, observation_spec, action_spec)
    actor_with_exploration, exploration_module = add_exploration(config, actor, action_spec)

    warmup(temp_env, actor_with_exploration, qvalue)
    temp_env.close()

    buffer = make_buffer(config)
    collector = make_async_collector(config, make_env_fn, actor_with_exploration, buffer)

    loss_module = make_loss_module(config, actor, qvalue, action_spec)
    optimizer_actor, optimizer_critic = make_optimizers(config, loss_module)
    target_net_updater = make_target_updater(config, loss_module)

    total_train_steps = 0

    window_size = config.data.frames_per_batch
    recent_losses = deque(maxlen=window_size)

    logger.info("Training started")

    train_config = config.train

    try:
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
                    
                    total_loss = loss_qvalue_val
                    if loss_actor_val is not None:
                        total_loss += loss_actor_val

                    recent_losses.append(total_loss)

                exploration_module.step(train_config.batch_size)
                
                avg_loss = sum(recent_losses) / len(recent_losses)
                train_writer.add_scalar("Loss", avg_loss, total_train_steps)
                
                total_train_steps += 1

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        collector.async_shutdown()
        train_writer.close()


if __name__ == "__main__":
    main()

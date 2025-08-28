import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from nn_laser_stabilizer.utils import set_seeds
from nn_laser_stabilizer.agents.td3 import (
    make_td3_agent,
    add_exploration,
    make_loss_module,
    make_optimizers,
    make_target_updater,
    train_step,
    warmup
)
from nn_laser_stabilizer.envs.utils import make_simulated_env, add_logger_to_env
from nn_laser_stabilizer.data.utils import make_buffer, make_sync_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir
from logging import getLogger
logger = getLogger(__name__)
from hydra.core.hydra_config import HydraConfig

CONFIG_NAME = "train_simulation"

@hydra.main(config_path=find_configs_dir(), config_name=CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    env_log_dir = os.path.join(hydra_output_dir, "env_logs")
    os.makedirs(env_log_dir, exist_ok=True)

    env = make_simulated_env(config)
    env = add_logger_to_env(env, env_log_dir)

    train_log_dir = os.path.join(hydra_output_dir, "train_logs")
    os.makedirs(train_log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=train_log_dir)

    action_spec = env.action_spec_unbatched
    observation_spec = env.observation_spec_unbatched["observation"]

    actor, qvalue = make_td3_agent(config, observation_spec, action_spec)
    actor_with_exploration, exploration_module = add_exploration(config, actor, action_spec)

    warmup_start_time = time.time()
    warmup(env, actor_with_exploration, qvalue)
    warmup_duration = time.time() - warmup_start_time
    logger.info(f"Warmup completed in {warmup_duration:.4f} seconds")

    buffer = make_buffer(config)
    collector = make_sync_collector(config, env, actor_with_exploration)
    collector_iterator = iter(collector) 

    loss_module = make_loss_module(config, actor, qvalue, action_spec)
    optimizer_actor, optimizer_critic = make_optimizers(config, loss_module)
    target_net_updater = make_target_updater(config, loss_module)

    train_config = config.train

    total_collected_frames = 0
    total_train_steps = 0
    total_training_time = 0
    total_iterations = 0

    recent_qvalue_losses = deque(maxlen=train_config.update_to_data)
    recent_actor_losses = deque(maxlen=train_config.update_to_data // train_config.update_actor_freq)

    try: 
        logger.info("Training started")
        training_start_time = time.time()

        while True:
            try:
                collector_start_time = time.time()
                tensordict_data = next(collector_iterator)
                collector_duration = time.time() - collector_start_time
                num_frames = tensordict_data.numel()
                time_per_frame = collector_duration / num_frames if num_frames > 0 else 0
                total_collected_frames += num_frames
                logger.info(f"Data collection step completed in {collector_duration:.4f} seconds, {time_per_frame:.6f} seconds per frame ({num_frames} frames)")

                buffer.extend(tensordict_data.unsqueeze(0).to_tensordict())

                exploration_module.step(tensordict_data.numel())

                if total_train_steps < train_config.total_train_steps:
                    for i in range(train_config.update_to_data):
                        batch = buffer.sample(train_config.batch_size)
                        step_start_time = time.time()
                        
                        update_actor = i % train_config.update_actor_freq == 0
                        loss_qvalue_val, loss_actor_val = train_step(
                            batch, loss_module, optimizer_actor, optimizer_critic, 
                            target_net_updater, update_actor
                        )

                        step_duration = time.time() - step_start_time
                        logger.info(f"Training step {total_train_steps * train_config.update_to_data + i + 1} completed in {step_duration:.4f} seconds")
                        
                        recent_qvalue_losses.append(loss_qvalue_val)
                        if loss_actor_val is not None:
                            recent_actor_losses.append(loss_actor_val)
                        
                        total_training_time += step_duration
                        total_iterations += 1
                    
                    avg_qvalue_loss = sum(recent_qvalue_losses) / len(recent_qvalue_losses)
                    train_writer.add_scalar("Loss/Critic", avg_qvalue_loss, total_train_steps)

                    avg_actor_loss = sum(recent_actor_losses) / len(recent_actor_losses)
                    train_writer.add_scalar("Loss/Actor", avg_actor_loss, total_train_steps)

                    total_train_steps += 1
            except StopIteration:
                logger.info("Collector iterator exhausted")
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception as ex:
        logger.warning(f"Error while training: {ex}")

    finally:
        total_loop_duration = time.time() - training_start_time
        logger.info(f"Training finished")
        logger.info(f"Total training loop duration: {total_loop_duration:.4f} seconds")
        if total_iterations > 0:
            avg_step_time = total_training_time / total_iterations
            logger.info(f"Average training step time: {avg_step_time:.4f} seconds")

        collector.shutdown()
        env.close()
        train_writer.close()

if __name__ == "__main__":
    main()

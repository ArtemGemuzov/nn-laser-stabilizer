import os
from collections import deque

import torch

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.logging.utils import (
    set_seeds
)
from nn_laser_stabilizer.logging.file_logger import SimpleFileLogger
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
from nn_laser_stabilizer.envs.utils import make_simulated_env, make_real_env, make_specs, add_logger_to_env, close_real_env, wrap_with_logger, transform_env
from nn_laser_stabilizer.data.utils import make_buffer, make_async_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir

from logging import getLogger
logger = getLogger(__name__)

from hydra.core.hydra_config import HydraConfig

CONFIG_NAME = "train_real"

@hydra.main(config_path=find_configs_dir(), config_name=CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    if not config.env.get("setpoint"):
        config.env.setpoint = float(input("Введите значение setpoint для среды: "))
    logger.info(f"Setpoint = {config.env.setpoint}")

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    env_log_dir = os.path.join(hydra_output_dir, "env_logs")
    os.makedirs(env_log_dir, exist_ok=True)

    def make_env(config, log_dir):
        env = make_real_env(config)
        env_with_logger = wrap_with_logger(env, log_dir=log_dir)
        return transform_env(config, env_with_logger)

    # TODO: aSyncDataCollector внутри себя создает фейковое окружение, поэтому при первом вызове нужно вернуть симуляцию окружения
    def make_env_factory(config, log_dir):
        first_call = True

        def env_fn():
            nonlocal first_call
            if first_call:
                first_call = False
                return make_simulated_env(config)
            return make_env(config, log_dir)

        return env_fn

    make_env_fn = make_env_factory(config, env_log_dir)

    train_log_dir = os.path.join(hydra_output_dir, "train_logs")
    os.makedirs(train_log_dir, exist_ok=True)
    train_logger = SimpleFileLogger(log_dir=train_log_dir)

    specs = make_specs(config.env.bounds)
    action_spec = specs["action"]
    observation_spec = specs["observation"]

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
                
                avg_qvalue_loss = sum(recent_qvalue_losses) / len(recent_qvalue_losses)
                avg_actor_loss = sum(recent_actor_losses) / len(recent_actor_losses)
                
                train_logger.log(f"step={total_train_steps} Loss/Critic={avg_qvalue_loss}; Loss/Actor={avg_actor_loss}")
                
                total_train_steps += 1

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception as ex:
        logger.warning(f"Error while training: {ex}")

    finally:
        logger.info("Training finished")

        model_save_dir = os.path.join(hydra_output_dir, "saved_models")
        os.makedirs(model_save_dir, exist_ok=True)

        actor_path = os.path.join(model_save_dir, "actor.pth")
        torch.save(actor.state_dict(), actor_path)

        qvalue_path = os.path.join(model_save_dir, "qvalue.pth")
        torch.save(qvalue.state_dict(), qvalue_path)
        
        collector.async_shutdown()
        train_logger.close()


if __name__ == "__main__":
    main()
import os
from collections import deque

import torch

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.logging.utils import (
    set_seeds
)
from nn_laser_stabilizer.agents.td3 import (
    make_td3_agent,
    make_loss_module,
    make_optimizers,
    make_target_updater,
    train_step,
    warmup_from_specs
)
from nn_laser_stabilizer.envs.utils import make_real_env, make_specs
from nn_laser_stabilizer.data.utils import make_buffer, make_async_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir
from nn_laser_stabilizer.logging.async_file_logger import AsyncFileLogger
from nn_laser_stabilizer.config.paths import get_hydra_output_dir

from logging import getLogger
logger = getLogger(__name__)

CONFIG_NAME = "train_real"

@hydra.main(config_path=find_configs_dir(), config_name=CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    if not config.env.get("setpoint"):
        config.env.setpoint = float(input("Введите значение setpoint для среды: "))
    logger.info(f"Setpoint = {config.env.setpoint}")

    output_dir = get_hydra_output_dir()

    # TODO: aSyncDataCollector внутри себя создает фейковое окружение, поэтому при первом вызове нужно вернуть симуляцию окружения
    def make_env_factory(config, output_dir):
        first_call = True

        def env_fn():
            nonlocal first_call
            if not first_call:
                return make_real_env(config, output_dir=output_dir)
            
            first_call = False
            from copy import deepcopy
            config_for_mocked_env = deepcopy(config)
            if not config_for_mocked_env.serial.get("use_mock"):
                config_for_mocked_env.serial.use_mock = True
            return make_real_env(config_for_mocked_env, output_dir=output_dir)

        return env_fn

    make_env_fn = make_env_factory(config, output_dir)

    train_log_dir = get_hydra_output_dir("train_logs")
    train_logger = AsyncFileLogger(log_dir=train_log_dir, filename="train.log")

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
        logger.info("Training process initiated")
        collector.start()
        
        min_data_required = config.data['min_data_for_training']
        training_started = False 
        
        while total_train_steps < train_config.total_train_steps:
            if len(buffer) > train_config.batch_size and len(buffer) >= min_data_required:
                if not training_started:
                    logger.info("Training started") 
                    training_started = True

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
                
                train_logger.log(f"step={total_train_steps} Loss/Critic={avg_qvalue_loss} Loss/Actor={avg_actor_loss}")
                
                total_train_steps += 1

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception as ex:
        logger.warning(f"Error while training: {ex}")

    finally:
        logger.info("Training finished")
        logger.info(f"Final buffer size: {len(buffer)} samples")

        model_save_dir = get_hydra_output_dir("saved_models")

        actor_path = os.path.join(model_save_dir, "actor.pth")
        torch.save(actor.state_dict(), actor_path)

        qvalue_path = os.path.join(model_save_dir, "qvalue.pth")
        torch.save(qvalue.state_dict(), qvalue_path)
        
        buffer_save_dir = get_hydra_output_dir("saved_data")
        buffer_path = os.path.join(buffer_save_dir, "replay_buffer.pkl")
        os.makedirs(buffer_save_dir, exist_ok=True)
        
        logger.info(f"Saving replay buffer with {len(buffer)} samples to {buffer_path}")
        buffer.dump(buffer_path)
        logger.info("Replay buffer saved successfully")
        
        collector.async_shutdown()
        train_logger.close()


if __name__ == "__main__":
    main()
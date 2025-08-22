import numpy as np

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.train_utils import (
    set_seeds,
    plot_results
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
from nn_laser_stabilizer.envs.utils import make_env
from nn_laser_stabilizer.data.utils import make_buffer, make_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir, DEFAULE_CONFIG_NAME

from logging import getLogger
logger = getLogger(__name__)


@hydra.main(config_path=find_configs_dir(), config_name=DEFAULE_CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)
    
    env = make_env(config)
    action_spec = env.action_spec_unbatched

    actor, qvalue = make_td3_agent(action_spec, config)
    actor, exploration_module = add_exploration(actor, action_spec, config)

    warmup(env, actor, qvalue)

    collector = make_collector(env, actor, config)
    buffer = make_buffer(config)
    loss_module = make_loss_module(actor, qvalue, action_spec, config)
    optimizer_actor, optimizer_critic = make_optimizers(loss_module, config)
    target_net_updater = make_target_updater(loss_module, config)

    total_collected_frames = 0
    losses = []
    rewards = []
    kp_log, ki_log, kd_log = [], [], []
    x_log, sp_log = [], []

    logger.info("Training started")

    try: 
        for tensordict_data in collector:
            observation = tensordict_data["observation"]
            x_log.extend(observation[:, 0].tolist())
            sp_log.extend(observation[:, 2].tolist())

            action = tensordict_data["action"]
            kp_log.extend(action[:, 0].tolist())
            ki_log.extend(action[:, 1].tolist())
            kd_log.extend(action[:, 2].tolist())
            
            buffer.extend(tensordict_data)
            total_collected_frames += tensordict_data.numel()

            current_reward = tensordict_data["next", "reward"].mean().item()
            rewards.append(current_reward)

            exploration_module.step(tensordict_data.numel())

            if total_collected_frames < config.max_train_steps:
                if len(buffer) >= config.batch_size:
                    for i in range(config.update_to_data):
                        batch = buffer.sample(batch_size=config.batch_size)
                        
                        loss_qvalue_val, loss_actor_val = train_step(
                            batch, loss_module, optimizer_actor, optimizer_critic, 
                            target_net_updater, i, config
                        )
                        
                        if loss_actor_val is not None:
                            losses.append(loss_actor_val + loss_qvalue_val)
                        else:
                            losses.append(loss_qvalue_val)

            logger.info(
                f"Frame {total_collected_frames}, "
                f"Average Reward: {np.mean(rewards):.8f}"
            )
            
            if len(losses) != 0:
                logger.info(f"Loss: {np.mean(losses):.8f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        collector.shutdown()
        env.close()

        logger.info("Training finished, plotting results...")
        plot_results(kp_log, ki_log, kd_log, x_log, sp_log)


if __name__ == "__main__":
    main()

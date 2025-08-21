import numpy as np

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.train_utils import (
    Config,
    set_seeds,
    make_env,
    make_td3_agent,
    make_collector,
    make_buffer,
    make_loss_module,
    make_optimizers,
    make_target_updater,
    train_step,
    plot_results
)
from nn_laser_stabilizer.find_configs_dir import find_configs_dir, DEFAULE_CONFIG_NAME

from logging import getLogger
logger = getLogger(__name__)


@hydra.main(config_path=find_configs_dir(), config_name=DEFAULE_CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    config = Config(**config)
    
    set_seeds(config.seed)
    
    env = make_env(config)
    model, actor_model_explore, exploration_module = make_td3_agent(env, config)
    collector = make_collector(env, actor_model_explore, config)
    buffer = make_buffer(config)
    
    action_spec = env.action_spec_unbatched.to(config.device)
    loss_module = make_loss_module(model, action_spec, config)
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
            collector.update_policy_weights_()

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
                        batch = buffer.sample(batch_size=config.batch_size).to(config.device)
                        
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

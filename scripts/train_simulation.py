from pathlib import Path

from collections import deque
from statistics import mean

from torch.utils.tensorboard import SummaryWriter

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
from nn_laser_stabilizer.envs.pid_tuning_experimental_env import to_pid_params, to_oscillator_params

from nn_laser_stabilizer.data.utils import make_buffer, make_collector
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir, DEFAULE_CONFIG_NAME

from logging import getLogger
logger = getLogger(__name__)

from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path=find_configs_dir(), config_name=DEFAULE_CONFIG_NAME, version_base=None)
def main(config: DictConfig) -> None:
    set_seeds(config.seed)

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=hydra_output_dir)

    env = make_env(config)
    action_spec = env.action_spec_unbatched
    observation_spec = env.observation_spec_unbatched["observation"]

    actor, qvalue = make_td3_agent(config, observation_spec, action_spec)
    actor_with_exploration, exploration_module = add_exploration(config, actor, action_spec)

    warmup(env, actor_with_exploration, qvalue)

    buffer = make_buffer(config)
    collector = make_collector(config, env, actor_with_exploration, buffer)
    loss_module = make_loss_module(config, actor, qvalue, action_spec)
    optimizer_actor, optimizer_critic = make_optimizers(config, loss_module)
    target_net_updater = make_target_updater(config, loss_module)

    total_collected_frames = 0
    window_size = config.data.frames_per_batch
    recent_rewards = deque(maxlen=window_size)
    recent_losses = deque(maxlen=window_size)

    kp_log, ki_log, kd_log = [], [], []
    x_log, sp_log = [], []

    logger.info("Training started")

    train_config = config.train

    try: 
        for tensordict_data in collector:
            observation = tensordict_data["observation"]
            x_log.extend(observation[:, 0].tolist())
            sp_log.extend(observation[:, 2].tolist())

            action = tensordict_data["action"]
            kp_log.extend(action[:, 0].tolist())
            ki_log.extend(action[:, 1].tolist())
            kd_log.extend(action[:, 2].tolist())
            
            total_collected_frames += tensordict_data.numel()
            buffer.extend(tensordict_data.unsqueeze(0).to_tensordict())

            current_reward = tensordict_data["next", "reward"].mean().item()
            recent_rewards.append(current_reward)

            exploration_module.step(tensordict_data.numel())

            if total_collected_frames < train_config.total_train_steps:
                for i in range(train_config.update_to_data):
                    batch = buffer.sample(train_config.batch_size)
                    
                    update_actor = i % train_config.update_actor_freq == 0
                    loss_qvalue_val, loss_actor_val = train_step(
                        batch, loss_module, optimizer_actor, optimizer_critic, 
                        target_net_updater, update_actor
                    )
                    
                    if loss_actor_val is not None:
                        recent_losses.append(loss_actor_val + loss_qvalue_val)
                    else:
                        recent_losses.append(loss_qvalue_val)

            avg_reward = mean(recent_rewards)
            logger.info(
                f"Frame {total_collected_frames}, "
                f"Recent Average Reward: {avg_reward:.8f}"
            )
            
            if len(recent_losses) != 0:
                avg_loss = mean(recent_losses)
                logger.info(f"Recent Loss: {avg_loss:.8f}")
            else:
                avg_loss = None

            # writer.add_scalar("Reward", avg_reward, total_collected_frames)
            # if avg_loss is not None:
            #     writer.add_scalar("Loss", avg_loss, total_collected_frames)

            # if step % 10 == 0:
            #     writer.add_scalar("Action/kp_mean", np.mean(kp_log[-window_size:]), total_collected_frames)
            #     writer.add_scalar("Action/ki_mean", np.mean(ki_log[-window_size:]), total_collected_frames)
            #     writer.add_scalar("Action/kd_mean", np.mean(kd_log[-window_size:]), total_collected_frames)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        collector.shutdown()
        env.close()

        logger.info("Training finished, plotting results...")
        plot_results(kp_log, ki_log, kd_log, x_log, sp_log)


if __name__ == "__main__":
    main()

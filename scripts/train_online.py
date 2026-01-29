from typing import cast
from functools import partial
from pathlib import Path
import time

import numpy as np

from nn_laser_stabilizer.data.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.data.sampler import make_sampler_from_config
from nn_laser_stabilizer.collector.collector import AsyncCollector, SyncCollector
from nn_laser_stabilizer.envs.env_wrapper import make_env_from_config, make_spaces_from_config
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.policy import make_policy_from_config
from nn_laser_stabilizer.optimizer import Optimizer
from nn_laser_stabilizer.experiment.decorator import experiment, ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger, PrefixedLogger
from nn_laser_stabilizer.model.actor import make_actor_from_config
from nn_laser_stabilizer.model.critic import make_critic_from_config
from nn_laser_stabilizer.algorithm.utils import make_updater_from_config


def validate(
    policy: Policy,
    env_factory,
    num_steps: int = 100,
) -> np.ndarray:
    policy.eval()
    env = env_factory()
    
    rewards = []
    obs, _ = env.reset()
    options = {}  
    
    for _ in range(num_steps):
        action, options = policy.act(obs, options) 
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        
        if done:
            options = {}
            obs, _ = env.reset()
    
    env.close()
    policy.train()
    
    return np.array(rewards)


@experiment("neural_controller")
def main(context: ExperimentContext):
    TRAIN_LOG_PREFIX = "TRAIN"
    
    context.logger.log("Creating components...")

    train_log_dir = Path(context.config.training.log_dir)
    train_logger = PrefixedLogger(
        logger=SyncFileLogger(log_dir=train_log_dir, log_file=context.config.training.log_file),
        prefix=TRAIN_LOG_PREFIX
    )
    
    is_async = context.config.collector.is_async
    
    env_factory = partial(make_env_from_config, env_config=context.config.env, seed=context.seed)
    
    observation_space, action_space = make_spaces_from_config(context.config.env, seed=context.seed)
    observation_dim = observation_space.dim
    action_dim = action_space.dim
    
    buffer = ReplayBuffer(
        capacity=context.config.buffer.capacity,
        obs_dim=observation_dim,
        action_dim=action_dim,
    )    
    sampler = make_sampler_from_config(
        buffer=buffer,
        sampler_config=context.config.sampler
    )

    network_config = context.config.network
    
    actor = make_actor_from_config(
        action_space=action_space,
        observation_space=observation_space,
        network_config=network_config,
    ).train()
    
    critic = make_critic_from_config(
        obs_dim=observation_dim,
        action_dim=action_dim,
        network_config=network_config,
    ).train()

    policy = make_policy_from_config(
        actor=actor,
        exploration_config=context.config.exploration,
    ).train()

    updater_cfg = context.config.updater
    updater = make_updater_from_config(
        updater_config=updater_cfg,
        actor=actor,
        critic=critic,
        actor_optimizer_factory=lambda params: Optimizer(
            params,
            lr=updater_cfg.actor_lr,
        ),
        critic_optimizer_factory=lambda params: Optimizer(
            params,
            lr=updater_cfg.critic_lr,
        ),
    )
    
    if is_async:
        context.logger.log("Starting async collector...")
        collector = AsyncCollector(
            buffer=buffer,
            policy=policy,
            env_factory=env_factory,
            seed=context.seed,
        )
    else:
        context.logger.log("Creating synchronous collector...")
        collector_env = make_env_from_config(context.config.env)
        collector = SyncCollector(
            buffer=buffer,
            env=collector_env,
            policy=policy,
        )

    num_steps = context.config.training.num_steps
    infinite_steps = num_steps == -1
    
    log_frequency = context.config.training.log_frequency
    logging_enabled = log_frequency > 0

    validation_frequency = context.config.validation.frequency
    validation_num_steps = context.config.validation.num_steps
    validation_enabled = validation_num_steps > 0 and validation_frequency > 0

    testing_num_steps = context.config.testing.num_steps
    testing_enabled = testing_num_steps > 0

    env_config = context.config.env
    if is_async:
        sync_frequency = context.config.collector.sync_frequency
    else:
        collect_steps_per_iteration = context.config.collector.collect_steps_per_iteration

    train_start_step = context.config.training.train_start_step
    sync_start_step = context.config.training.sync_start_step
    
    with collector:
        try:
            if is_async:
                context.logger.log("Collector started. Waiting for data accumulation...")
            else:
                context.logger.log("Collector started. Initial data collection...")
            
            collector.collect(train_start_step)
            
            if not is_async:
                context.logger.log(f"Initial data collection completed. Buffer size: {len(buffer)}")
            
            context.logger.log("Training started")

            step = 0
            while infinite_steps or step < num_steps:
                step += 1

                if not is_async:
                    collector.collect(collect_steps_per_iteration)
                
                batch = sampler.sample()

                loss_q1, loss_q2, actor_loss = updater.update_step(batch)
                
                if is_async and step >= sync_start_step and step % sync_frequency == 0:
                    cast(AsyncCollector, collector).sync()
                    
                if logging_enabled and step % log_frequency == 0:
                    timestamp = time.time()
                    if actor_loss is not None:
                        train_logger.log(
                            f"step: actor_loss={actor_loss} buffer_size={len(buffer)} "
                            f"loss_q1={loss_q1} loss_q2={loss_q2} step={step} time={timestamp}"
                        )
                    else:
                        train_logger.log(
                            f"step: buffer_size={len(buffer)} "
                            f"loss_q1={loss_q1} loss_q2={loss_q2} step={step} time={timestamp}"
                        )
                
                if validation_enabled and step % validation_frequency == 0:
                    rewards = validate(
                        policy,
                        lambda: make_env_from_config(env_config),
                        num_steps=validation_num_steps,
                    )
                    train_logger.log(
                        f"validation: episodes={rewards.size} reward_mean={rewards.mean()} "
                        f"reward_sum={rewards.sum()} step={step} time={time.time()}"
                    )
            
            if testing_enabled:
                context.logger.log("Testing...")
                test_rewards = validate(policy, lambda: make_env_from_config(env_config), num_steps=testing_num_steps)
                train_logger.log(
                    f"testing: episodes={test_rewards.size} reward_mean={test_rewards.mean()} "
                    f"reward_sum={test_rewards.sum()} time={time.time()}"
                )
            
            context.logger.log("Training completed.")
            context.logger.log(f"Final buffer size: {len(buffer)}")
        finally:
            context.logger.log("Saving models...")
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            updater.actor.save(models_dir / "actor.pth")
            updater.critic1.save(models_dir / "critic1.pth")
            updater.critic2.save(models_dir / "critic2.pth")
            updater.actor_target.save(models_dir / "actor_target.pth")
            updater.critic1_target.save(models_dir / "critic1_target.pth")
            updater.critic2_target.save(models_dir / "critic2_target.pth")
            context.logger.log(f"Models saved to {models_dir}")
            
            context.logger.log("Saving replay buffer...")
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            buffer.save(data_dir / "replay_buffer.pth")
            context.logger.log(f"Replay buffer saved to {data_dir}")
    
    train_logger.close()
    context.logger.log("Collector stopped.")


if __name__ == "__main__":
    main()


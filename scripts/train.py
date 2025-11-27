from functools import partial
from pathlib import Path
import time

import numpy as np

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.sampler import make_sampler_from_config
from nn_laser_stabilizer.collector import AsyncCollector, SyncCollector
from nn_laser_stabilizer.env_wrapper import make_env_from_config
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.policy import make_policy
from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.training import td3_train_step
from nn_laser_stabilizer.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.experiment import experiment, ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger
from nn_laser_stabilizer.actor import make_actor_from_config
from nn_laser_stabilizer.critic import make_critic_from_config


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


@experiment("pid_delta_tuning")
def main(context: ExperimentContext):
    context.logger.log("Creating components...")

    train_log_dir = Path(context.config.training.log_dir)
    train_logger = SyncFileLogger(log_dir=train_log_dir, log_file=context.config.training.log_file)
    
    is_async = context.config.collector.is_async
    
    env_factory = partial(make_env_from_config, env_config=context.config.env, seed=context.seed)
    env = env_factory()
    
    observation_space = env.observation_space
    observation_dim = observation_space.dim

    action_space = env.action_space
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
    
    exploration_steps = context.config.training.exploration_steps
    policy_factory = partial(
        make_policy,
        actor=actor,
        action_space=action_space,
        exploration_steps=exploration_steps
    )

    policy = policy_factory().train()
    
    critic = make_critic_from_config(
        obs_dim=observation_dim,
        action_dim=action_dim,
        network_config=network_config,
    ).train()
    
    loss_module = TD3Loss(
        actor=actor,
        critic=critic,
        action_space=action_space,
        gamma=context.config.loss.gamma,
        policy_noise=context.config.loss.policy_noise,
        noise_clip=context.config.loss.noise_clip,
    )
    
    actor_optimizer = Optimizer(loss_module.actor.parameters(), lr=context.config.optimizer.actor_lr)
    critic_optimizer = Optimizer(
        list(loss_module.critic1.parameters()) + list(loss_module.critic2.parameters()),
        lr=context.config.optimizer.critic_lr
    )
    soft_updater = SoftUpdater(loss_module, tau=context.config.optimizer.tau)
    
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
    
    with collector:
        try:
            if is_async:
                context.logger.log("Collector started. Waiting for data accumulation...")
            else:
                context.logger.log("Collector started. Initial data collection...")
            
            collector.collect(context.config.training.initial_collect_steps)
            
            if not is_async:
                context.logger.log(f"Initial data collection completed. Buffer size: {len(buffer)}")
            
            context.logger.log("Training started")
            
            num_training_steps = context.config.training.num_training_steps
            policy_freq = context.config.training.policy_freq
            log_frequency = context.config.training.log_frequency
            validation_frequency = context.config.validation.frequency
            env_config = context.config.env
            validation_num_steps = context.config.validation.num_steps
            testing_num_steps = context.config.testing.num_steps
            
            if is_async:
                sync_frequency = context.config.collector.sync_frequency
            else:
                collect_steps_per_iteration = context.config.collector.collect_steps_per_iteration
            
            for step in range(1, num_training_steps + 1):
                if not is_async:
                    collector.collect(collect_steps_per_iteration)
                
                batch = sampler.sample()
                
                update_actor_and_target = (step % policy_freq == 0)
                
                loss_q1, loss_q2, actor_loss = td3_train_step(
                    batch,
                    loss_module,
                    critic_optimizer=critic_optimizer,
                    actor_optimizer=actor_optimizer,
                    soft_updater=soft_updater,
                    update_actor_and_target=update_actor_and_target,
                )
                
                if is_async and step % sync_frequency == 0:
                    collector.sync()
                    
                if step % log_frequency == 0:
                    timestamp = time.time()
                    if actor_loss is not None:
                        train_logger.log(f"step={step} time={timestamp:.6f} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} actor_loss={actor_loss:.4f} buffer_size={len(buffer)}")
                    else:
                        train_logger.log(f"step={step} time={timestamp:.6f} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} buffer_size={len(buffer)}")
                
                if validation_num_steps > 0 and step % validation_frequency == 0:
                    rewards = validate(policy, lambda: make_env_from_config(env_config), num_steps=validation_num_steps)
                    train_logger.log(f"validation step={step} time={time.time():.6f} reward_sum={rewards.sum():.4f} reward_mean={rewards.mean():.4f} episodes={rewards.size}")
            
            if testing_num_steps > 0:
                context.logger.log("Testing...")
                test_rewards = validate(policy, lambda: make_env_from_config(env_config), num_steps=testing_num_steps)
                train_logger.log(f"testing time={time.time():.6f} reward_sum={test_rewards.sum():.4f} reward_mean={test_rewards.mean():.4f} episodes={test_rewards.size}")
            
            context.logger.log("Training completed.")
            context.logger.log(f"Final buffer size: {len(buffer)}")
        finally:
            context.logger.log("Saving models...")
            models_dir = Path("models").mkdir(parents=True, exist_ok=True)
            loss_module.actor.save(models_dir / "actor.pth")
            loss_module.critic1.save(models_dir / "critic1.pth")
            loss_module.critic2.save(models_dir / "critic2.pth")
            loss_module.actor_target.save(models_dir / "actor_target.pth")
            loss_module.critic1_target.save(models_dir / "critic1_target.pth")
            loss_module.critic2_target.save(models_dir / "critic2_target.pth")
            context.logger.log(f"Models saved to {models_dir}")
            
            context.logger.log("Saving replay buffer...")
            data_dir = Path("data").mkdir(parents=True, exist_ok=True)
            buffer.save(data_dir / "replay_buffer.pth")
            context.logger.log(f"Replay buffer saved to {data_dir}")
    
    train_logger.close()
    context.logger.log("Collector stopped.")


if __name__ == "__main__":
    main()


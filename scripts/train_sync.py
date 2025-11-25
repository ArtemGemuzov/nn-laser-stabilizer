from pathlib import Path

import numpy as np

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.collector import SyncCollector
from nn_laser_stabilizer.env_wrapper import make_env_from_config
from nn_laser_stabilizer.sampler import BatchSampler
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.actor import MLPActor
from nn_laser_stabilizer.critic import MLPCritic
from nn_laser_stabilizer.policy import make_policy
from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.training import td3_train_step
from nn_laser_stabilizer.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.experiment import ExperimentContext, experiment
from nn_laser_stabilizer.logger import SyncFileLogger


def validate(
    policy: Policy,
    env_factory,
    num_steps: int = 100,
) -> np.ndarray:
    policy.eval()
    env = env_factory()
    
    rewards = []
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        
        if done:
            obs, _ = env.reset()
    
    env.close()
    policy.train()
    
    return np.array(rewards)

@experiment("pendulum")
def main(context: ExperimentContext):
    context.console_logger.log("Creating components...")

    train_log_dir = Path(context.config.training.log_dir)
    train_logger = SyncFileLogger(log_dir=train_log_dir, log_file=context.config.training.log_file)
    
    env = make_env_from_config(context.config.env, seed=context.seed)
    
    observation_space = env.observation_space
    observation_dim = observation_space.dim

    action_space = env.action_space
    action_dim = action_space.dim
    
    buffer = ReplayBuffer(
        capacity=context.config.buffer.capacity,
        obs_dim=observation_dim,
        action_dim=action_dim,
    )
    
    sampler = BatchSampler(buffer=buffer, batch_size=context.config.sampler.batch_size)
    
    actor = MLPActor(
        obs_dim=observation_dim,
        action_dim=action_dim,
        action_space=action_space,
        hidden_sizes=tuple(context.config.network.hidden_sizes),
    ).train()
    
    exploration_steps = context.config.training.get.exploration_steps
    policy = make_policy(actor=actor, action_space=action_space, exploration_steps=exploration_steps).train()
    
    critic = MLPCritic(
        obs_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(context.config.network.hidden_sizes),
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
    
    collector_env = make_env_from_config(context.config.env)
    
    context.console_logger.log("Creating synchronous collector...")
    
    with SyncCollector(
        buffer=buffer,
        env=collector_env,
        policy=policy,  
    ) as collector:
        context.console_logger.log("Collector started. Initial data collection...")
        
        collector.collect(context.config.training.initial_collect_steps)
        context.console_logger.log(f"Initial data collection completed. Buffer size: {len(buffer)}")
        
        context.console_logger.log(f"Training started")
        
        num_training_steps = context.config.training.num_training_steps
        collect_steps_per_iteration = context.config.training.collect_steps_per_iteration
        policy_freq = context.config.training.policy_freq
        log_frequency = context.config.training.log_frequency
        validation_frequency = context.config.training.validation_frequency
        env_config = context.config.env
        validation_num_steps = context.config.validation.num_steps
        final_validation_num_steps = context.config.validation.final_num_steps
        
        for step in range(num_training_steps):
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
            
            if step % log_frequency == 0:
                if actor_loss is not None:
                    train_logger.log(f"step={step} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} actor_loss={actor_loss:.4f} buffer_size={len(buffer)}")
                else:
                    train_logger.log(f"step={step} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} buffer_size={len(buffer)}")
            
            if step % validation_frequency == 0 and step > 0:
                rewards = validate(actor, lambda: make_env_from_config(env_config), num_steps=validation_num_steps)
                train_logger.log(f"validation step={step} reward_sum={rewards.sum():.4f} reward_mean={rewards.mean():.4f} episodes={rewards.size}")
        
        context.console_logger.log("Final validation...")
        final_rewards = validate(actor, lambda: make_env_from_config(context.config.env), num_steps=final_validation_num_steps)
        train_logger.log(f"final_validation reward_sum={final_rewards.sum():.4f} reward_mean={final_rewards.mean():.4f} episodes={final_rewards.size}")
        
        context.console_logger.log("Saving models...")
        models_dir = context.models_dir
        actor.save(models_dir / "actor.pth")
        loss_module.critic1.save(models_dir / "critic1.pth")
        context.console_logger.log(f"Models saved to {models_dir}")
        
        context.console_logger.log("Saving replay buffer...")
        buffer.save(context.data_dir / "replay_buffer.pth")
        context.console_logger.log(f"Replay buffer saved to {context.data_dir}")
        
        context.console_logger.log("Training completed.")
        context.console_logger.log(f"Final buffer size: {len(buffer)}")
    
    train_logger.close()
    context.console_logger.log("Collector stopped.")


if __name__ == "__main__":
    main()


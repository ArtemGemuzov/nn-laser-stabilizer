from functools import partial
from pathlib import Path

import numpy as np

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.collector import AsyncCollector
from nn_laser_stabilizer.env_wrapper import make_env, make_env_from_config
from nn_laser_stabilizer.sampler import BatchSampler
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.actor import MLPActor
from nn_laser_stabilizer.critic import MLPCritic
from nn_laser_stabilizer.policy_wrapper import RandomExplorationPolicy
from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.training import td3_train_step
from nn_laser_stabilizer.optimizer import Optimizer, SoftUpdater
from nn_laser_stabilizer.experiment import experiment, ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger


def make_policy(action_space, observation_space, hidden_sizes, exploration_steps: int = 0) -> Policy:
    actor = MLPActor(
        obs_dim=observation_space.dim,
        action_dim=action_space.dim,
        action_space=action_space,
        hidden_sizes=hidden_sizes,
    )
    
    if exploration_steps > 0:
        return RandomExplorationPolicy(
            actor=actor,
            exploration_steps=exploration_steps,
            action_space=action_space
        )
    return actor


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


@experiment("pid_delta_tuning")
def main(context: ExperimentContext):
    print("Creating components...")

    train_log_dir = Path(context.config.training.log_dir)
    train_logger = SyncFileLogger(log_dir=train_log_dir, log_file=context.config.training.log_file)
    
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
    
    sampler = BatchSampler(buffer=buffer, batch_size=context.config.sampler.batch_size)
    
    exploration_steps = context.config.training.get("exploration_steps", 0)
    
    policy_factory = partial(
        make_policy, 
        action_space=action_space, 
        observation_space=observation_space,
        hidden_sizes=tuple(context.config.network.hidden_sizes),
        exploration_steps=exploration_steps
    )
   
    policy = policy_factory().train()
    actor = policy.actor if isinstance(policy, RandomExplorationPolicy) else policy
    
    critic = MLPCritic(
        obs_dim=observation_dim, 
        action_dim=action_dim, 
        hidden_sizes=tuple(context.config.network.hidden_sizes)
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
    
    print("Starting collector...")
    
    with AsyncCollector(
        buffer=buffer,
        policy=policy,
        env_factory=env_factory,
    ) as collector:
        print("Collector started. Waiting for data accumulation...")
        
        collector.collect(context.config.training.initial_collect_steps)
        print(f"Training started. Buffer size: {len(buffer)}")
        
        num_training_steps = context.config.training.num_training_steps
        sync_frequency = context.config.training.sync_frequency
        policy_freq = context.config.training.policy_freq
        log_frequency = context.config.training.log_frequency
        validation_frequency = context.config.training.validation_frequency
        env_config = context.config.env
        validation_num_steps = context.config.validation.num_steps
        final_validation_num_steps = context.config.validation.final_num_steps
        
        for step in range(num_training_steps):
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
            
            if step % sync_frequency == 0:
                collector.sync()
                
            if step % log_frequency == 0:
                if actor_loss is not None:
                    log_line = f"step={step} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} actor_loss={actor_loss:.4f} buffer_size={len(buffer)}"
                    train_logger.log(log_line)
                    print(f"Step {step}: loss_q1={loss_q1:.4f}, loss_q2={loss_q2:.4f}, "
                            f"actor_loss={actor_loss:.4f}, buffer size={len(buffer)}")
                else:
                    log_line = f"step={step} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} buffer_size={len(buffer)}"
                    train_logger.log(log_line)
                    print(f"Step {step}: loss_q1={loss_q1:.4f}, loss_q2={loss_q2:.4f}, "
                            f"buffer size={len(buffer)}")
            
            if step % validation_frequency == 0 and step > 0:
                rewards = validate(actor, lambda: make_env_from_config(env_config), num_steps=validation_num_steps)
                log_line = f"validation step={step} reward_sum={rewards.sum():.4f} reward_mean={rewards.mean():.4f} episodes={rewards.size}"
                train_logger.log(log_line)
                print(f"Validation (step {step}): reward = {rewards.sum():.4f} for {rewards.size} episodes")
        
        print("\nFinal validation...")
        final_rewards = validate(actor, lambda: make_env_from_config(env_config), num_steps=final_validation_num_steps)
        log_line = f"final_validation reward_sum={final_rewards.sum():.4f} reward_mean={final_rewards.mean():.4f} episodes={final_rewards.size}"
        train_logger.log(log_line)
        print(f"Final average reward: {final_rewards.mean()}")
        
        print("Saving models...")
        models_dir = context.models_dir
        actor.save(models_dir / "actor.pth")
        loss_module.critic1.save(models_dir / "critic1.pth")
        print(f"Models saved to {models_dir}")
        
        print("Saving replay buffer...")
        buffer.save(context.data_dir / "replay_buffer.pth")
        print(f"Replay buffer saved to {context.data_dir}")
        
        print("Training completed.")
        print(f"Final buffer size: {len(buffer)}")
    
    train_logger.close()
    print("Collector stopped.")


if __name__ == "__main__":
    main()
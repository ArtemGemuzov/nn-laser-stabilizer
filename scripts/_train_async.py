from functools import partial

import numpy as np

import torch
import torch.optim as optim

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.collector import AsyncCollector
from nn_laser_stabilizer.env_wrapper import make_env
from nn_laser_stabilizer.sampler import BatchSampler
from nn_laser_stabilizer.policy import Policy
from nn_laser_stabilizer.actor import MLPActor
from nn_laser_stabilizer.critic import MLPCritic
from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.training import td3_train_step
from nn_laser_stabilizer.utils import SoftUpdater
from nn_laser_stabilizer.experiment import experiment, ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger


def make_policy(action_space, observation_space) -> MLPActor:
    return MLPActor(
        obs_dim=observation_space.dim,
        action_dim=action_space.dim,
        action_space=action_space,
        hidden_sizes=(400, 300),
    )


def validate(
    policy: Policy,
    env_factory,
    num_steps: int = 100,
) -> np.ndarray:
    policy.eval()
    env = env_factory()
    
    rewards = []
    obs, _ = env.reset()
    
    with torch.no_grad():
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


@experiment("_train_sync_async.yaml")
def main(context: ExperimentContext):
    print("Creating components...")

    train_logger = SyncFileLogger(log_dir=context.logs_dir, log_file="train.log")
    
    env = make_env("Pendulum-v1")
    
    observation_space = env.observation_space
    observation_dim = observation_space.dim

    action_space = env.action_space
    action_dim = action_space.dim
    
    buffer = ReplayBuffer(
        capacity=100000,
        obs_dim=observation_dim,
        action_dim=action_dim,
    )
    
    sampler = BatchSampler(buffer=buffer, batch_size=64)
    
    policy_factory = partial(make_policy, action_space=action_space, observation_space=observation_space)
   
    actor = policy_factory().train()
    critic = MLPCritic(
        obs_dim=observation_dim, 
        action_dim=action_dim, 
        hidden_sizes=(400, 300)
    ).train()
    
    loss_module = TD3Loss(
        actor=actor,
        critic=critic,
        action_space=action_space,
        gamma=0.98,
        policy_noise=0.1,
        noise_clip=0.25,
    )
    
    actor_optimizer = optim.Adam(loss_module.actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(
        list(loss_module.critic1.parameters()) + list(loss_module.critic2.parameters()),
        lr=1e-3
    )
    
    soft_updater = SoftUpdater(loss_module, tau=0.005)
    
    print("Starting collector...")
    
    with AsyncCollector(
        buffer=buffer,
        env_factory=lambda: make_env("Pendulum-v1"),
        policy_factory=policy_factory,
    ) as collector:
        print("Collector started. Waiting for data accumulation...")
        
        initial_collect_steps = 10000
        collector.collect(initial_collect_steps)
        print(f"Training started. Buffer size: {len(buffer)}")
        
        num_training_steps = 20000
        sync_frequency = 100 
        log_frequency = 100
        validation_frequency = 500
        policy_freq = 2
        
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
                collector.sync(actor)
                
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
                rewards = validate(actor, lambda: make_env("Pendulum-v1"), num_steps=200)
                log_line = f"validation step={step} reward_sum={rewards.sum():.4f} reward_mean={rewards.mean():.4f} episodes={rewards.size}"
                train_logger.log(log_line)
                print(f"Validation (step {step}): reward = {rewards.sum():.4f} for {rewards.size} episodes")
        
        print("\nFinal validation...")
        final_rewards = validate(actor, lambda: make_env("Pendulum-v1"), num_steps=1000)
        log_line = f"final_validation reward_sum={final_rewards.sum():.4f} reward_mean={final_rewards.mean():.4f} episodes={final_rewards.size}"
        train_logger.log(log_line)
        print(f"Final average reward: {final_rewards.mean()}")
        
        print("Training completed.")
        print(f"Final buffer size: {len(buffer)}")
    
    train_logger.close()
    print("Collector stopped.")


if __name__ == "__main__":
    main()
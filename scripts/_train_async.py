from functools import partial

import numpy as np

import gymnasium as gym

import torch
import torch.optim as optim

from nn_laser_stabilizer.replay_buffer import SharedReplayBuffer
from nn_laser_stabilizer.collector import AsyncCollector
from nn_laser_stabilizer.env import TorchEnvWrapper, PendulumNoVelEnv, PidDeltaTuningEnv
from nn_laser_stabilizer.policy import Policy, MLPPolicy
from nn_laser_stabilizer.critic import MLPCritic
from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.training import td3_train_step
from nn_laser_stabilizer.utils import SoftUpdater


def make_gym_env():
    env = gym.make("Pendulum-v1")
    return TorchEnvWrapper(env)


def make_pid_env() -> TorchEnvWrapper:
    return PidDeltaTuningEnv.create(
        setpoint=1200.0,
        warmup_steps=100,
        block_size=50,
        pretrain_blocks=10,
        burn_in_steps=10,
        use_logging=True,
        log_dir=".",
    )


def make_env():
    return make_gym_env()


def make_policy(action_space, observation_space) -> MLPPolicy:
    return MLPPolicy(
        obs_dim=observation_space.shape[0],
        action_dim=action_space.shape[0],
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


def main():
    print("Создание компонентов...")
    
    env = make_env()
    observation_space = env.unwrapped.observation_space
    observation_dim = observation_space.shape[0]

    action_space = env.unwrapped.action_space
    action_dim = action_space.shape[0]
    
    buffer = SharedReplayBuffer(
        capacity=100000,
        obs_shape=observation_space.shape,
        action_shape=action_space.shape,
    )
    
    policy_factory = partial(make_policy, action_space=action_space, observation_space=observation_space)
   
    actor = policy_factory().train()
    critic = MLPCritic(obs_dim=observation_dim, action_dim=action_dim, hidden_sizes=(256, 256)).train()
    
    loss_module = TD3Loss(
        actor=actor,
        critic=critic,
        action_space=action_space,
        gamma=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
    )
    
    actor_optimizer = optim.Adam(loss_module.actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(
        list(loss_module.critic1.parameters()) + list(loss_module.critic2.parameters()),
        lr=1e-3
    )
    
    soft_updater = SoftUpdater(loss_module, tau=0.005)
    
    print("Запуск коллектора...")
    
    with AsyncCollector(
        buffer=buffer,
        env_factory=make_env,
        policy_factory=policy_factory,
    ) as collector:
        print("Коллектор запущен. Ожидание накопления данных...")
        
        initial_collect_steps = 10000
        collector.collect(initial_collect_steps)
        print(f"Начало обучения. Размер буфера: {len(buffer)}")
        
        num_training_steps = 20000
        sync_frequency = 100 
        log_frequency = 100
        validation_frequency = 500 
        policy_freq = 2  
        
        for step in range(num_training_steps):
            batch = buffer.sample(batch_size=64)
            
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
                    print(f"Шаг {step}: loss_q1={loss_q1:.4f}, loss_q2={loss_q2:.4f}, "
                            f"actor_loss={actor_loss:.4f}, размер буфера={len(buffer)}")
                else:
                    print(f"Шаг {step}: loss_q1={loss_q1:.4f}, loss_q2={loss_q2:.4f}, "
                            f"размер буфера={len(buffer)}")
            
            if step % validation_frequency == 0 and step > 0:
                rewards = validate(actor, make_gym_env, num_steps=200)
                print(f"Валидация (шаг {step}): награда = {rewards.sum():.4f} за {rewards.size} эпизодов")
        
        print("\nФинальная валидация...")
        final_rewards = validate(actor, make_gym_env, num_steps=1000)
        print(f"Финальная средняя награда: {final_rewards.mean()}")
        
        print("Обучение завершено.")
        print(f"Финальный размер буфера: {len(buffer)}")
    
    print("Коллектор остановлен.")


if __name__ == "__main__":
    main()
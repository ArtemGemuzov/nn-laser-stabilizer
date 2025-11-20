from functools import partial

import numpy as np

import gymnasium as gym

import torch
import torch.optim as optim

from nn_laser_stabilizer.replay_buffer import ReplayBuffer
from nn_laser_stabilizer.collector import AsyncCollector
from nn_laser_stabilizer.env import TorchEnvWrapper, PendulumNoVelEnv, PidDeltaTuningEnv
from nn_laser_stabilizer.policy import Policy, MLPPolicy
from nn_laser_stabilizer.critic import MLPCritic
from nn_laser_stabilizer.loss import TD3Loss
from nn_laser_stabilizer.training import td3_train_step
from nn_laser_stabilizer.utils import SoftUpdater
from nn_laser_stabilizer.experiment import experiment, ExperimentContext
from nn_laser_stabilizer.logger import SyncFileLogger


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


@experiment("_train_sync_async.yaml")
def main(context: ExperimentContext):
    config = context.config
    
    print(f"Эксперимент: {context.experiment_name}")
    print(f"Директория: {context.experiment_dir}")
    
    print("Создание компонентов...")

    logger = SyncFileLogger(log_dir=context.logs_dir, log_file="train.log")
    
    env = make_env()
    observation_space = env.unwrapped.observation_space
    observation_dim = observation_space.shape[0]

    action_space = env.unwrapped.action_space
    action_dim = action_space.shape[0]
    
    buffer = ReplayBuffer(
        capacity=config.data.buffer_size,
        obs_shape=observation_space.shape,
        action_shape=action_space.shape,
    )
    
    policy_factory = partial(make_policy, action_space=action_space, observation_space=observation_space)
   
    actor = policy_factory().train()
    critic = MLPCritic(
        obs_dim=observation_dim, 
        action_dim=action_dim, 
        hidden_sizes=tuple(config.agent.q_mlp_num_cells)
    ).train()
    
    loss_module = TD3Loss(
        actor=actor,
        critic=critic,
        action_space=action_space,
        gamma=config.agent.gamma,
        policy_noise=config.agent.policy_noise,
        noise_clip=config.agent.noise_clip,
    )
    
    actor_optimizer = optim.Adam(loss_module.actor.parameters(), lr=config.agent.learning_rate_actor)
    critic_optimizer = optim.Adam(
        list(loss_module.critic1.parameters()) + list(loss_module.critic2.parameters()),
        lr=config.agent.learning_rate_critic
    )
    
    soft_updater = SoftUpdater(loss_module, tau=config.agent.target_update_eps)
    
    print("Запуск коллектора...")
    
    with AsyncCollector(
        buffer=buffer,
        env_factory=make_env,
        policy_factory=policy_factory,
    ) as collector:
        print("Коллектор запущен. Ожидание накопления данных...")
        
        initial_collect_steps = config.data.min_data_for_training
        collector.collect(initial_collect_steps)
        print(f"Начало обучения. Размер буфера: {len(buffer)}")
        
        num_training_steps = config.train.total_train_steps
        sync_frequency = 100 
        log_frequency = config.validation.log_frequency
        validation_frequency = config.validation.validation_frequency
        policy_freq = config.train.update_actor_freq
        
        for step in range(num_training_steps):
            batch = buffer.sample(batch_size=config.train.batch_size)
            
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
                    logger.log(log_line)
                    print(f"Шаг {step}: loss_q1={loss_q1:.4f}, loss_q2={loss_q2:.4f}, "
                            f"actor_loss={actor_loss:.4f}, размер буфера={len(buffer)}")
                else:
                    log_line = f"step={step} loss_q1={loss_q1:.4f} loss_q2={loss_q2:.4f} buffer_size={len(buffer)}"
                    logger.log(log_line)
                    print(f"Шаг {step}: loss_q1={loss_q1:.4f}, loss_q2={loss_q2:.4f}, "
                            f"размер буфера={len(buffer)}")
            
            if step % validation_frequency == 0 and step > 0:
                rewards = validate(actor, make_gym_env, num_steps=config.validation.validation_steps)
                log_line = f"validation step={step} reward_sum={rewards.sum():.4f} reward_mean={rewards.mean():.4f} episodes={rewards.size}"
                logger.log(log_line)
                print(f"Валидация (шаг {step}): награда = {rewards.sum():.4f} за {rewards.size} эпизодов")
        
        print("\nФинальная валидация...")
        final_rewards = validate(actor, make_gym_env, num_steps=config.validation.final_validation_steps)
        log_line = f"final_validation reward_sum={final_rewards.sum():.4f} reward_mean={final_rewards.mean():.4f} episodes={final_rewards.size}"
        logger.log(log_line)
        print(f"Финальная средняя награда: {final_rewards.mean()}")
        
        print("Обучение завершено.")
        print(f"Финальный размер буфера: {len(buffer)}")
    
    logger.close()
    print("Коллектор остановлен.")


if __name__ == "__main__":
    main()
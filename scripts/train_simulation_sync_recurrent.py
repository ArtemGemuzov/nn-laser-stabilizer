import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import hydra
from omegaconf import DictConfig

from nn_laser_stabilizer.recurrent.utils import PendulumNoVelWrapper, FastRecurrentReplayBuffer, collect_data_episode
from nn_laser_stabilizer.recurrent.td3 import RecurrentTD3
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir

@hydra.main(config_path=find_configs_dir(), config_name="train_simulation_recurrent", version_base=None)
def main(config: DictConfig):
    env = gym.make(config.env.name)
    env = PendulumNoVelWrapper(env)

    np.random.seed(config.env.seed)
    torch.manual_seed(config.env.seed)
    env.reset(seed=config.env.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    device = torch.device("cpu")

    agent = RecurrentTD3(
        obs_dim=obs_dim,
        action_dim=action_dim,
        mlp_hidden_size=config.agent.mlp_hidden_size,
        rnn_hidden_size=config.agent.rnn_hidden_size,
        num_rnn_layers=config.agent.num_rnn_layers,
        max_action=max_action,
        min_action=min_action,
        lr=config.agent.lr,
        gamma=config.agent.gamma,
        tau=config.agent.tau,
        policy_noise=config.agent.policy_noise,
        noise_clip=config.agent.noise_clip,
        policy_freq=config.agent.policy_freq,
        shared_summary=config.shared_summary,
        device=device
    )

    replay_buffer = FastRecurrentReplayBuffer(obs_dim=obs_dim, action_dim=action_dim)

    episode_rewards = []

    print("Начинаем обучение...")

    try:
        for episode in range(config.env.num_episodes):
            rewards = collect_data_episode(
                env, agent, replay_buffer=replay_buffer, num_steps=config.env.num_steps
            )
            episode_mean_reward = np.mean(rewards)
            episode_rewards.append(episode_mean_reward)

            if len(replay_buffer) > config.agent.batch_size * config.agent.seq_len:
                for _ in range(config.agent.update_to_data):
                    agent.train_step(
                        replay_buffer,
                        batch_size=config.agent.batch_size,
                        seq_len=config.agent.seq_len,
                    )

            print(f"Эпизод {episode}, Средняя награда: {episode_mean_reward:.2f}")

    except KeyboardInterrupt:
        print("Обучение прервано пользователем")

    finally:
        env.close()

        window_size = 10

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title("Средняя награда за эпизод")
        plt.xlabel("Эпизод")
        plt.ylabel("Награда")

        plt.subplot(1, 2, 2)
        smoothed_rewards = [
            np.mean(episode_rewards[max(0, i - window_size): i + 1])
            for i in range(len(episode_rewards))
        ]
        plt.plot(smoothed_rewards)
        plt.title(f"Скользящее среднее наград (окно {window_size})")
        plt.xlabel("Эпизод")
        plt.ylabel("Средняя награда")

        plt.tight_layout()
        plt.show()

        print("Обучение завершено")


if __name__ == "__main__":
    main()
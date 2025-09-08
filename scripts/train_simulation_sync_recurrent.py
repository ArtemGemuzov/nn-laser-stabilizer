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
def main(cfg: DictConfig):
    env = gym.make(cfg.env.name)
    env = PendulumNoVelWrapper(env)

    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    env.reset(seed=cfg.env.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    device = torch.device("cpu")

    agent = RecurrentTD3(
        obs_dim=obs_dim,
        action_dim=action_dim,
        mlp_hidden_size=cfg.agent.mlp_hidden_size,
        rnn_hidden_size=cfg.agent.rnn_hidden_size,
        num_rnn_layers=cfg.agent.num_rnn_layers,
        max_action=max_action,
        min_action=min_action,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        policy_noise=cfg.agent.policy_noise,
        noise_clip=cfg.agent.noise_clip,
        policy_freq=cfg.agent.policy_freq,
        device=device
    )

    replay_buffer = FastRecurrentReplayBuffer(obs_dim=obs_dim, action_dim=action_dim)

    episode_rewards = []

    print("Начинаем обучение...")

    try:
        for episode in range(cfg.env.num_episodes):
            rewards = collect_data_episode(
                env, agent, replay_buffer=replay_buffer, num_steps=cfg.env.num_steps
            )
            episode_mean_reward = np.mean(rewards)
            episode_rewards.append(episode_mean_reward)

            if len(replay_buffer) > cfg.agent.batch_size * cfg.agent.seq_len:
                for _ in range(cfg.agent.update_to_data):
                    agent.train_step(
                        replay_buffer,
                        batch_size=cfg.agent.batch_size,
                        seq_len=cfg.agent.seq_len,
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
import os
import time
import torch
import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from nn_laser_stabilizer.recurrent.utils import PendulumNoVelWrapper, FastRecurrentReplayBuffer
from nn_laser_stabilizer.recurrent.td3 import RecurrentTD3
from nn_laser_stabilizer.config.find_configs_dir import find_configs_dir

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Simple3x3Env(gym.Env):
    def __init__(self):
        super(Simple3x3Env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.state = np.zeros(3, dtype=np.float32)

        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        info = {}
        return self.state, info

    def step(self, action):
        self.current_step += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)

        noise = np.random.normal(0, 0.05, size=(3,))
        self.state = np.clip(self.state + action + noise, -10.0, 10.0)

        reward = -np.sum(np.square(self.state))

        terminated = np.any(np.abs(self.state) > 9.5)
        truncated = self.current_step >= self.max_steps

        info = {}
        return self.state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

@hydra.main(config_path=find_configs_dir(), config_name="train_simulation_recurrent", version_base=None)
def benchmark_recurrent(cfg: DictConfig) -> None:
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(hydra_output_dir, exist_ok=True)

    env = Simple3x3Env()

    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    env.reset(seed=cfg.env.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

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
        device=device,
    )

    replay_buffer = FastRecurrentReplayBuffer(obs_dim=obs_dim, action_dim=action_dim)

    print("Собираем данные для буфера...")
    observation, _ = env.reset()
    for _ in range(200):  
        action = env.action_space.sample()
        next_observation, reward, _, _, _ = env.step(action)

        replay_buffer.push(observation, action, reward, next_observation)
        observation = next_observation

    print(f"Размер буфера: {len(replay_buffer)}")

    observation, _ = env.reset()
    observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # === Warmup ===
    warmup_steps = 1_000
    print(f"Выполняем {warmup_steps} warmup-шагов...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = agent.select_action(observation_tensor)

    n_steps = 20_000
    action_times = []

    print("Измеряем время на вычисление действия...")
    for _ in range(n_steps):
        start = time.perf_counter()
        with torch.no_grad():
            _ = agent.select_action(observation_tensor)
        action_times.append(time.perf_counter() - start)

    action_times = np.array(action_times)
    print(f"[Policy] Mean: {action_times.mean():.8f} | Std: {action_times.std():.8f} | "
          f"Min: {action_times.min():.8f} | Max: {action_times.max():.8f}")

    print("Измеряем время одного train_step...")
    train_times = []
    n_train_steps = 500

    for _ in range(n_train_steps):
        start = time.perf_counter()
        agent.train_step(
            replay_buffer,
            batch_size=cfg.agent.batch_size,
            seq_len=cfg.agent.seq_len
        )
        train_times.append(time.perf_counter() - start)

    train_times = np.array(train_times)
    print(f"[Train step] Mean: {train_times.mean():.8f} | Std: {train_times.std():.8f} | "
          f"Min: {train_times.min():.8f} | Max: {train_times.max():.8f}")


if __name__ == "__main__":
    benchmark_recurrent()

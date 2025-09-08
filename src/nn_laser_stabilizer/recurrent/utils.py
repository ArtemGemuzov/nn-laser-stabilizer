import torch
import numpy as np
import gymnasium as gym
    
from nn_laser_stabilizer.recurrent.td3 import RecurrentTD3

class FastRecurrentReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=100_000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.capacity = capacity
        self.size = 0
        self.index = 0
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)

    def push(self, observation, action, reward, next_observation):
        # TODO: циклическая вставка может ломать формирование последовательностей
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_observations[self.index] = next_observation
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, seq_len, device='cpu'):
        if self.size < seq_len:
            raise ValueError("Buffer too small for sequence length")
        
        valid_indices = np.arange(self.size - seq_len + 1)
        start_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        
        indices = start_indices[:, np.newaxis] + np.arange(seq_len + 1)
        
        observations_batch = self.observations[indices]
        actions_batch = self.actions[indices[:, :-1]]
        rewards_batch = self.rewards[indices[:, :-1]]
        
        observations_tensor = torch.from_numpy(observations_batch).to(device)
        actions_tensor = torch.from_numpy(actions_batch).to(device)
        rewards_tensor = torch.from_numpy(rewards_batch).to(device)
        
        return observations_tensor, actions_tensor, rewards_tensor

    def __len__(self):
        return self.size


def collect_data_episode(env: gym.Env, agent: RecurrentTD3, replay_buffer: FastRecurrentReplayBuffer, num_steps=1000):
    agent.reset_hidden()
    observation, _ = env.reset()

    rewards = []
    for _ in range(num_steps):
        action = agent.select_action(observation)
        next_observation, reward, _, _, _ = env.step(action)

        replay_buffer.push(observation.copy(), action.copy(), reward, next_observation.copy())

        observation = next_observation
        rewards.append(reward)
        
    return rewards


class PendulumNoVelWrapper(gym.ObservationWrapper):
    """
    Обертка для Pendulum-v1, которая убирает угловую скорость из наблюдений.
    Возвращает только cos(theta) и sin(theta).
    """
    def __init__(self, env):
        super(PendulumNoVelWrapper, self).__init__(env)

        low = np.delete(self.observation_space.low, 2)
        high = np.delete(self.observation_space.high, 2)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[:2]
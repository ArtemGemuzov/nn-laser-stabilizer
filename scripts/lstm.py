import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt

class TanhScaler(nn.Module):
    def __init__(self, min_val: float, max_val: float):
        super(TanhScaler, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(x)
        return self.min_val + (x + 1) * (self.max_val - self.min_val) / 2
    
class RecurrentReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size, seq_len, device='cpu'):
        """Сэмплирует последовательность и возвращает тензоры"""
        sequences = []
        max_start = len(self.buffer) - seq_len
        start_indices = [random.randint(0, max_start) for _ in range(batch_size)]
        
        for start_idx in start_indices:
            seq = [self.buffer[start_idx + i] for i in range(seq_len)]
            sequences.append(seq)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        
        for seq in sequences:
            seq_obs = []
            seq_actions = []
            seq_rewards = []
            seq_next_obs = []
            
            for transition in seq:
                seq_obs.append(transition[0])
                seq_actions.append(transition[1])
                seq_rewards.append(transition[2])
                seq_next_obs.append(transition[3])
            
            observations.append(seq_obs)
            actions.append(seq_actions)
            rewards.append(seq_rewards)
            next_observations.append(seq_next_obs)
        
        observations = torch.FloatTensor(observations).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_observations = torch.FloatTensor(next_observations).to(device)
        
        return observations, actions, rewards, next_observations

    def __len__(self):
        return len(self.buffer)

class Summarizer(nn.Module):
    def __init__(self, input_dim, hidden_size=32):
        super(Summarizer, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        
    def forward(self, observation, hidden=None):
        if len(observation.shape) == 2:
            observation = observation.unsqueeze(1)
        lstm_out, hidden = self.lstm(observation, hidden)
        return lstm_out, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)

class MLPTanhActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256, max_action=1.0, min_action=-1.0):
        super(MLPTanhActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.scaler = TanhScaler(min_val=min_action, max_val=max_action)

    def forward(self, x):
        return self.scaler(self.net(x))

class MLPCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256):
        super(MLPCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state_summary, action):
        sa = torch.cat([state_summary, action], dim=-1)
        return self.net(sa)

class RecurrentTD3:
    def __init__(self, obs_dim, action_dim, max_action=1.0, min_action=-1.0,
                 lr=1e-3, gamma=0.99, tau=0.05, policy_noise=0.2, noise_clip=0.5, 
                 policy_freq=2, device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.total_it = 0
        
        self.actor_summarizer = Summarizer(obs_dim).to(device)
        self.actor_mlp = MLPTanhActor(self.actor_summarizer.hidden_size, action_dim, 
                                    max_action=max_action, min_action=min_action).to(device)
        
        self.actor_summarizer_target = Summarizer(obs_dim).to(device)
        self.actor_mlp_target = MLPTanhActor(self.actor_summarizer.hidden_size, action_dim,
                                           max_action=max_action, min_action=min_action).to(device)
        
        self.actor_summarizer_target.load_state_dict(self.actor_summarizer.state_dict())
        self.actor_mlp_target.load_state_dict(self.actor_mlp.state_dict())
        
        self.critic1_summarizer = Summarizer(obs_dim).to(device)
        self.critic1_mlp = MLPCritic(self.critic1_summarizer.hidden_size, action_dim).to(device)
        
        self.critic1_summarizer_target = Summarizer(obs_dim).to(device)
        self.critic1_mlp_target = MLPCritic(self.critic1_summarizer.hidden_size, action_dim).to(device)
        
        self.critic1_summarizer_target.load_state_dict(self.critic1_summarizer.state_dict())
        self.critic1_mlp_target.load_state_dict(self.critic1_mlp.state_dict())
        
        self.critic2_summarizer = Summarizer(obs_dim).to(device)
        self.critic2_mlp = MLPCritic(self.critic2_summarizer.hidden_size, action_dim).to(device)
        
        self.critic2_summarizer_target = Summarizer(obs_dim).to(device)
        self.critic2_mlp_target = MLPCritic(self.critic2_summarizer.hidden_size, action_dim).to(device)
        
        self.critic2_summarizer_target.load_state_dict(self.critic2_summarizer.state_dict())
        self.critic2_mlp_target.load_state_dict(self.critic2_mlp.state_dict())
        
        self.actor_optimizer = optim.Adam(
            list(self.actor_summarizer.parameters()) + list(self.actor_mlp.parameters()), lr=lr
        )
        self.critic1_optimizer = optim.Adam(
            list(self.critic1_summarizer.parameters()) + list(self.critic1_mlp.parameters()), lr=lr
        )
        self.critic2_optimizer = optim.Adam(
            list(self.critic2_summarizer.parameters()) + list(self.critic2_mlp.parameters()), lr=lr
        )
    
        self.actor_hidden = None
        self.reset_hidden_states()
    
    def reset_hidden_states(self):
        self.actor_hidden = self.actor_summarizer.init_hidden(device=self.device)
    
    def select_action(self, observation):
        observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            summary, self.actor_hidden = self.actor_summarizer(observation, self.actor_hidden)
            action = self.actor_mlp(summary.squeeze(1))
            action = action.squeeze(0)
        
        return action.cpu().numpy()
    
    def train(self, replay_buffer, batch_size=32, seq_len=10):
        self.total_it += 1
        
        observations, actions, rewards, next_observations = replay_buffer.sample(
            batch_size, seq_len, self.device
        )

        with torch.no_grad():
            target_actor_hidden = self.actor_summarizer_target.init_hidden(batch_size, self.device)
            target_summary, _ = self.actor_summarizer_target(next_observations, target_actor_hidden)
            next_actions = self.actor_mlp_target(target_summary)
            
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(self.min_action, self.max_action)
            
            target_critic1_hidden = self.critic1_summarizer_target.init_hidden(batch_size, self.device)
            target_critic1_summary, _ = self.critic1_summarizer_target(next_observations, target_critic1_hidden)
            target_q1 = self.critic1_mlp_target(target_critic1_summary, next_actions)
            
            target_critic2_hidden = self.critic2_summarizer_target.init_hidden(batch_size, self.device)
            target_critic2_summary, _ = self.critic2_summarizer_target(next_observations, target_critic2_hidden)
            target_q2 = self.critic2_mlp_target(target_critic2_summary, next_actions)
            
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + self.gamma * target_q
        
        critic1_hidden = self.critic1_summarizer.init_hidden(batch_size, self.device)
        critic1_summary, _ = self.critic1_summarizer(observations, critic1_hidden)
        current_q1 = self.critic1_mlp(critic1_summary, actions)
        
        critic2_hidden = self.critic2_summarizer.init_hidden(batch_size, self.device)
        critic2_summary, _ = self.critic2_summarizer(observations, critic2_hidden)
        current_q2 = self.critic2_mlp(critic2_summary, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            actor_hidden = self.actor_summarizer.init_hidden(batch_size, self.device)
            actor_summary, _ = self.actor_summarizer(observations, actor_hidden)
            actor_actions = self.actor_mlp(actor_summary)
            
            critic1_hidden = self.critic1_summarizer.init_hidden(batch_size, self.device)
            critic1_summary, _ = self.critic1_summarizer(observations, critic1_hidden)
            actor_q1 = self.critic1_mlp(critic1_summary, actor_actions)
            
            actor_loss = -actor_q1.mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            
            for param, target_param in zip(self.actor_summarizer.parameters(), self.actor_summarizer_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor_mlp.parameters(), self.actor_mlp_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic1_summarizer.parameters(), self.critic1_summarizer_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic1_mlp.parameters(), self.critic1_mlp_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic2_summarizer.parameters(), self.critic2_summarizer_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic2_mlp.parameters(), self.critic2_mlp_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic1_loss.item() + critic2_loss.item(), actor_loss.item() if actor_loss is not None else None


def collect_data_episode(env: gym.Env, agent: RecurrentTD3, replay_buffer: RecurrentReplayBuffer, max_steps=1000):
    agent.reset_hidden_states()
    observation, _ = env.reset()
    action = np.zeros(env.action_space.shape)

    total_reward = 0
    step = 0

    while step < max_steps:
        action = agent.select_action(observation)
        next_observation, reward, _, _, _ = env.step(action)

        replay_buffer.push(observation.copy(), action.copy(), reward, next_observation.copy())

        observation = next_observation
        total_reward += reward
        step += 1

    return total_reward


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

def main():
    env_name = "Pendulum-v1"
    max_episodes = 200
    max_steps = 200
    seq_len = 10
    batch_size = 128
    update_to_data = 16
    
    env = gym.make(env_name)
    # env = PendulumNoVelWrapper(env)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    agent = RecurrentTD3(state_dim, action_dim, max_action, min_action, device=device)
    replay_buffer = RecurrentReplayBuffer()
    
    episode_rewards = []
    
    print("Начинаем обучение...")
    
    for episode in range(max_episodes):
        episode_reward = collect_data_episode(env, agent, replay_buffer=replay_buffer, max_steps=max_steps)
        episode_rewards.append(episode_reward)
        
        if len(replay_buffer) > batch_size:
            for _ in range(update_to_data): 
                critic_loss, actor_loss = agent.train(replay_buffer, batch_size=batch_size, seq_len=seq_len)
       
        print(f"Эпизод {episode}, Средняя награда: {episode_reward / max_steps:.2f}")
    
    env.close()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Награда за эпизод')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    
    plt.subplot(1, 2, 2)

    window_size = 10
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-window_size):i+1]) 
                       for i in range(len(episode_rewards))]
    plt.plot(smoothed_rewards)
    plt.title(f'Скользящее среднее наград (окно {window_size})')
    plt.xlabel('Эпизод')
    plt.ylabel('Средняя награда')
    
    plt.tight_layout()
    plt.show()
    
    print("Обучение завершено!")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
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
        
        indices = start_indices[:, np.newaxis] + np.arange(seq_len)
        
        observations_batch = self.observations[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_observations_batch = self.next_observations[indices]
        
        observations_tensor = torch.from_numpy(observations_batch).to(device)
        actions_tensor = torch.from_numpy(actions_batch).to(device)
        rewards_tensor = torch.from_numpy(rewards_batch).to(device)
        next_observations_tensor = torch.from_numpy(next_observations_batch).to(device)
        
        return observations_tensor, actions_tensor, rewards_tensor, next_observations_tensor

    def __len__(self):
        return self.size

class Summarizer(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=2):
        super(Summarizer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.hidden = None

    def forward(self, observation, hidden=None):
        if len(observation.shape) == 2:
            observation = observation.unsqueeze(1)
        lstm_out, hidden = self.lstm(observation, hidden)
        return lstm_out, hidden
    
    def reset_hidden(self):
        self.hidden = None

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

    def forward(self, observation_summary):
        return self.scaler(self.net(observation_summary))

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

    def forward(self, observation_summary, action):
        return self.net(torch.cat([observation_summary, action], dim=-1))

def soft_update(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class RecurrentTD3:
    def __init__(self, obs_dim, action_dim, 
                 mlp_hidden_size=256, rnn_hidden_size=32, num_rnn_layers=1, 
                 max_action=1.0, min_action=-1.0,
                 lr=1e-3, gamma=0.99, tau=0.05, policy_noise=0.2, noise_clip=0.5, 
                 policy_freq=2, device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.hidden_size = rnn_hidden_size
        self.num_layers = num_rnn_layers

        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.total_it = 0
        
        # TODO: отключить градиенты всем target сетям

        self.actor_summarizer = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.actor_mlp = MLPTanhActor(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size,
                                    max_action=max_action, min_action=min_action).to(device)
        
        self.actor_summarizer_target = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.actor_mlp_target = MLPTanhActor(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size,
                                           max_action=max_action, min_action=min_action).to(device)
        
        self.actor_summarizer_target.load_state_dict(self.actor_summarizer.state_dict())
        self.actor_mlp_target.load_state_dict(self.actor_mlp.state_dict())
        
        self.critic1_summarizer = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.critic1_mlp = MLPCritic(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size).to(device)
        
        self.critic1_summarizer_target = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.critic1_mlp_target = MLPCritic(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size).to(device)
        
        self.critic1_summarizer_target.load_state_dict(self.critic1_summarizer.state_dict())
        self.critic1_mlp_target.load_state_dict(self.critic1_mlp.state_dict())
        
        self.critic2_summarizer = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.critic2_mlp = MLPCritic(rnn_hidden_size, action_dim).to(device)
        
        self.critic2_summarizer_target = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.critic2_mlp_target = MLPCritic(rnn_hidden_size, action_dim).to(device)
        
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
    
        self.reset_hidden()

    def reset_hidden(self):
        self.actor_hidden = None
    
    def select_action(self, observation):
        observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            summary, self.actor_hidden = self.actor_summarizer(observation, self.actor_hidden)
            action = self.actor_mlp(summary.squeeze(1))
            action = action.squeeze(0)
        
        return action.cpu().numpy()
    
    def train_step(self, replay_buffer, batch_size=32, seq_len=10):
        self.total_it += 1
        
        observations, actions, rewards, next_observations = replay_buffer.sample(
            batch_size, seq_len, self.device
        )

        with torch.no_grad():
            target_summary, _ = self.actor_summarizer_target(next_observations)
            next_actions = self.actor_mlp_target(target_summary)
            
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(self.min_action, self.max_action)
            
            target_critic1_summary, _ = self.critic1_summarizer_target(next_observations)
            target_q1 = self.critic1_mlp_target(target_critic1_summary, next_actions)
            
            target_critic2_summary, _ = self.critic2_summarizer_target(next_observations)
            target_q2 = self.critic2_mlp_target(target_critic2_summary, next_actions)
            
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + self.gamma * target_q

        critic1_summary, _ = self.critic1_summarizer(observations)
        current_q1 = self.critic1_mlp(critic1_summary, actions)
        
        critic2_summary, _ = self.critic2_summarizer(observations)
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
            actor_summary, _ = self.actor_summarizer(observations)
            actor_actions = self.actor_mlp(actor_summary)
            
            critic1_summary, _ = self.critic1_summarizer(observations)
            actor_q1 = self.critic1_mlp(critic1_summary, actor_actions)
            
            actor_loss = -actor_q1.mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # TODO: использовать partial
            soft_update(self.actor_summarizer, self.actor_summarizer_target, self.tau)
            soft_update(self.actor_mlp, self.actor_mlp_target, self.tau)

            soft_update(self.critic1_summarizer, self.critic1_summarizer_target, self.tau)
            soft_update(self.critic1_mlp, self.critic1_mlp_target, self.tau)

            soft_update(self.critic2_summarizer, self.critic2_summarizer_target, self.tau)
            soft_update(self.critic2_mlp, self.critic2_mlp_target, self.tau)
        
        return critic1_loss.item() + critic2_loss.item(), actor_loss.item() if actor_loss is not None else None


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

def main():
    env_name = "Pendulum-v1"
    num_episodes = 500
    num_steps = 200

    seq_len = 20
    batch_size = 64
    update_to_data = 16
    
    env = gym.make(env_name)
    env = PendulumNoVelWrapper(env)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    agent = RecurrentTD3(obs_dim, action_dim, max_action=max_action, min_action=min_action, device=device)
    replay_buffer = FastRecurrentReplayBuffer(obs_dim=obs_dim, action_dim=action_dim)
    
    episode_rewards = []
    
    print("Начинаем обучение...")
    
    try:
        for episode in range(num_episodes):
            rewards = collect_data_episode(env, agent, replay_buffer=replay_buffer, num_steps=num_steps)
            episode_mean_reward = np.mean(rewards)
            episode_rewards.append(episode_mean_reward)
            
            if len(replay_buffer) > batch_size * seq_len:
                for _ in range(update_to_data): 
                    critic_loss, actor_loss = agent.train_step(replay_buffer, batch_size=batch_size, seq_len=seq_len)
        
            print(f"Эпизод {episode}, Средняя награда: {episode_mean_reward:.2f}")

    except KeyboardInterrupt:
        print("Обучение прервано")

    finally:
        env.close()

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Средняя награда за эпизод')
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
        
        print("Обучение завершено")

if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F

from functools import partial
import copy 
    
from nn_laser_stabilizer.recurrent.actor_critic import MLPTanhActor, MLPCritic
from nn_laser_stabilizer.recurrent.summarizer import Summarizer

def soft_update(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def make_target(model: torch.nn.Module) -> torch.nn.Module:
    target = copy.deepcopy(model)
    target.load_state_dict(model.state_dict())
    for param in target.parameters():
        param.requires_grad = False
    return target

def make_optimizer(*modules, lr=1e-3):
    params = []
    for module in modules:
        params += list(module.parameters())
    return torch.optim.Adam(params, lr=lr)

class RecurrentTD3:
    def __init__(self, obs_dim, action_dim, 
                 mlp_hidden_size=256, rnn_hidden_size=64, num_rnn_layers=2, 
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

        self.actor_mlp = MLPTanhActor(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size,
                                    max_action=max_action, min_action=min_action).to(device)
        self.actor_mlp_target = make_target(self.actor_mlp)
        
        self.actor_summarizer = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.actor_summarizer_target = make_target(self.actor_summarizer)

        self.critic1_mlp = MLPCritic(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size).to(device)
        self.critic1_mlp_target = make_target(self.critic1_mlp)

        self.critic1_summarizer = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.critic1_summarizer_target = make_target(self.critic1_summarizer)

        self.critic2_mlp = MLPCritic(rnn_hidden_size, action_dim, hidden_size=mlp_hidden_size).to(device)
        self.critic2_mlp_target = make_target(self.critic2_mlp)

        self.critic2_summarizer = Summarizer(obs_dim, rnn_hidden_size, num_rnn_layers).to(device)
        self.critic2_summarizer_target = make_target(self.critic2_summarizer)

        self.make_optimizer = partial(make_optimizer, lr=lr)
        self.actor_optimizer = self.make_optimizer(self.actor_summarizer, self.actor_mlp)
        self.critic1_optimizer = self.make_optimizer(self.critic1_summarizer, self.critic1_mlp)
        self.critic2_optimizer = self.make_optimizer(self.critic2_summarizer, self.critic2_mlp)

        self.soft_update = partial(soft_update, tau=self.tau)
    
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
                -self.noise_clip, self.noise_clip
            )
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
            actor_q1 = self.critic1_mlp(critic1_summary.detach(), actor_actions)

            actor_loss = -actor_q1.mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor_mlp, self.actor_mlp_target)
            self.soft_update(self.actor_summarizer, self.actor_summarizer_target)

            self.soft_update(self.critic1_summarizer, self.critic1_summarizer_target)
            self.soft_update(self.critic1_mlp, self.critic1_mlp_target)

            self.soft_update(self.critic2_summarizer, self.critic2_summarizer_target)
            self.soft_update(self.critic2_mlp, self.critic2_mlp_target)

        return critic1_loss.item() + critic2_loss.item(), actor_loss.item() if actor_loss is not None else None


